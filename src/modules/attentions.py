import torch
import torch.nn.functional as F

from src.modules.layers import Conv1d, Linear


def create_self_atten(
    name,
    memory_size,
    num_units,
    query_radius,
    smooth=False,
    scale_score_at_eval=1.0,
    **kwargs,
):
    qry_size = memory_size
    key_size = memory_size
    if name == "bah":
        return BahdanauAttention(
            num_units,
            qry_size,
            key_size,
            query_radius=query_radius,
            smooth=smooth,
            scale_score_at_eval=scale_score_at_eval,
        )
    else:
        raise NotImplementedError(f"attention module '{name}' is not implemented!")


class _Attention(torch.nn.Module):
    def __init__(
        self,
        num_units,
        query_size,
        key_size,
        value_size=None,
        same_kv=False,
        query_radius=1,
    ):
        super().__init__()
        self.qry_size = query_size
        self.qry_length = query_radius * 2 - 1
        self.key_size = key_size
        self.val_size = value_size or key_size
        self.num_units = num_units
        self._same_kv = same_kv
        # project query
        self._conv_query = Conv1d(
            query_size,
            query_size,
            self.qry_length,
            self.qry_length,
            bias=False,
            init_kwargs={"init_type": "xavier_normal"},
        )

    def forward(self, query, key, value=None):
        if value is None:  # default value from key
            value = key
        if self._same_kv:  # check same
            assert key.data_ptr() == value.data_ptr()
        assert (
            query.shape[1] == self.qry_length and query.shape[2] == self.qry_size
        ), "query should be in shape (N, {}, {}), but {}".format(self.qry_length, self.qry_size, query.shape)
        assert key.size(2) == self.key_size, "key should be in shape (N, T, {})".format(self.key_size)
        assert value.size(2) == self.val_size, "value should be in shape (N, T, {})".format(self.val_size)
        assert key.size(1) == value.size(1), "key, value has different length: {} != {}".format(
            key.size(1), value.size(1)
        )
        # project query
        query = self._conv_query(query.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        align = self.get_alignment(query, key)
        assert align.size(1) == query.size(1)
        assert align.size(2) == key.size(1)
        context = torch.bmm(align, value)
        return context, align

    def get_alignment(self, query, key):
        """
        Input:
            query: (N, T_q, C_q)
            key:   (N, T_k, C_k)
        Return:
            alignment: (N, T_q, T_k)
        """
        raise NotImplementedError()


def _smoothing_normalization(e, dim=-1):
    return torch.sigmoid(e) / torch.sum(torch.sigmoid(e), dim=dim, keepdim=True)


class BahdanauAttention(_Attention):
    def __init__(
        self,
        num_units,
        query_size,
        key_size,
        query_radius=1,
        smooth=False,
        scale_score_at_eval=1.0,
    ):
        super().__init__(
            num_units,
            query_size,
            key_size,
            query_radius=query_radius,
            value_size=None,
            same_kv=False,
        )
        self.score_scaling = scale_score_at_eval
        # projections
        self.proj_key = Linear(self.key_size, self.num_units, bias=False, init_kwargs={"init_type": "xavier_normal"})
        self.proj_qry = Linear(self.qry_size, self.num_units, bias=False, init_kwargs={"init_type": "xavier_normal"})
        self.v = Linear(self.num_units, 1, bias=False, init_kwargs={"init_type": "xavier_normal"})
        self.b = torch.nn.Parameter(torch.zeros((1, 1, self.num_units)))
        # normalize function
        if not smooth:
            self.normalize = torch.nn.functional.softmax
        else:
            self.normalize = _smoothing_normalization

    def _get_score(self, query, key):
        assert query.shape[1] == 1
        N, L, _ = key.shape
        b = self.b
        s = self.v(torch.tanh(query + key + b)).view(N, 1, L)
        return s

    def get_alignment(self, query, key):
        # project
        qry = self.proj_qry(query)
        key = self.proj_key(key)
        # get score
        score = self._get_score(query=qry, key=key)
        if not self.training:
            score = score * self.score_scaling
        # normalize score
        # align = self.normalize(score, dim=-1)
        align = F.gumbel_softmax(score, tau=1.0, hard=False)
        return align
