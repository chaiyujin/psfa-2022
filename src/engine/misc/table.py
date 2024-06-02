from copy import deepcopy
from textwrap import wrap
from typing import Any, List, Optional, Set, Tuple, Union

from omegaconf import DictConfig, ListConfig


class Table(object):
    DIVIDER_DOUBLE = "__DIVIDER_DOUBLE__"
    DIVIDER_SINGLE = "__DIVIDER_SINGLE__"

    def __init__(
        self,
        *metadata,
        alignment: Tuple[str, str] = ("left", "middle"),
        auto_divide: bool = True,
        list_mode: str = "compact",
        dict_mode: str = "multi-line",
        max_width=None,
        max_column_width=64,
        indent: str = "",
    ):
        self._indent: str = indent
        self._n_columns: int = len(metadata)
        self._rows: List[Optional[List[str]]] = []
        self._align_x = alignment[0]
        self._align_y = alignment[1]
        self._auto_divide = auto_divide
        self._list_mode = list_mode
        self._dict_mode = dict_mode
        self._max_width = None  # TODO: max_width for entire table
        self._max_column_width = max_column_width
        assert self._align_x in ("left", "center", "right")
        assert self._align_y in ("top", "middle", "bottom")
        assert self._list_mode in ["compact", "multi-line"]
        assert self._dict_mode in ["multi-line"]  # TODO: compact dict mode

        # * First row: metadata
        if len(metadata) > 0:
            self._rows.append(list(deepcopy(metadata)))
            self._rows.append(self.DIVIDER_DOUBLE)

    def add_row(self, *values):
        # * Update row
        columns = []
        for col, value in enumerate(values):
            columns.append(value)
        self._rows.append(columns)
        self._n_columns = max(self._n_columns, len(columns))
        if self._auto_divide:
            self._rows.append(self.DIVIDER_SINGLE)

    def divide(self, divider_type=DIVIDER_DOUBLE):
        if len(self._rows) > 0 and self._rows[-1] in [self.DIVIDER_DOUBLE, self.DIVIDER_SINGLE]:
            self._rows.pop(-1)
        self._rows.append(divider_type)

    def _is_divider(self, line):
        return line == self.DIVIDER_SINGLE or line == self.DIVIDER_DOUBLE

    def _divider_double(self, max_width_list):
        divider = "="
        for col_width in max_width_list:
            divider += "=" * col_width + "="
        return divider

    def _divider_single(self, max_width_list):
        divider = "+"
        for col_width in max_width_list:
            divider += "-" * col_width + "+"
        return divider

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                   To String                                                  * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def _to_strs(self, value) -> Tuple[str, ...]:
        assert value is None or isinstance(value, (int, float, str, list, tuple, dict, ListConfig, DictConfig))

        def _trim_empty_line(lines):
            while len(lines) > 0 and len(lines[0]) == 0:
                lines = lines[1:]
            while len(lines) > 0 and len(lines[-1]) == 0:
                lines = lines[:-1]
            return lines

        def _str(value):
            return " " + str(value) + " "  # extra padding

        if value is None:
            return ("None",)
        elif isinstance(value, (int, float)):
            return tuple([_str(x) for x in wrap(str(value), self._max_column_width)])
        elif isinstance(value, str):
            # * Process '\n' into multi-line
            lines = _trim_empty_line(value.split("\n"))
            lines = [_str(x) for x in lines]
            return tuple(lines)
        elif isinstance(value, (tuple, list, ListConfig)):
            if self._list_mode == "compact":
                lines = []
                for x in value:
                    lines.extend(self._to_strs(x))
                value = ", ".join(x.strip() for x in lines)
                value = wrap(value, width=self._max_column_width)
                return tuple([_str(x) for x in value])
            else:
                assert self._list_mode == "multi-line"
                # * Concat into multi-line
                lines = []
                for subval in value:
                    if len(lines) > 0:
                        lines.append(self.DIVIDER_SINGLE)  # inner cell divider
                    lines.extend(self._to_strs(subval))
                lines = _trim_empty_line(lines)
            return tuple(lines)
        elif isinstance(value, (dict, DictConfig)):
            # * Sub table
            sub_table = Table(
                alignment=(self._align_x, self._align_y), auto_divide=self._auto_divide, list_mode=self._list_mode
            )
            for key, val in value.items():
                sub_table.add_row(str(key), val)
            lines = str(sub_table).split("\n")
            # remove top and bottom divider
            lines = lines[1 : len(lines) - 1]
            # remove left and right bar
            lines = [x[1:-1] for x in lines]
            return tuple(_trim_empty_line(lines))
        else:
            return (str(value),)

    def __str__(self):
        # * We convert all cell into Tuple[str, ...], which can be multi-line.
        rows: List[Optional[List[Tuple[str, ...]]]] = []
        # * Also, get the max width for each column and max height for each row
        max_width_list = [0 for _ in range(self._n_columns)]
        max_height_list = [0 for _ in range(len(self._rows))]
        for i_row, row in enumerate(self._rows):
            if self._is_divider(row):
                rows.append(row)
            elif len(row) == 0:
                rows.append(self.DIVIDER_SINGLE)
            else:
                # values
                cols = [self._to_strs(value) for value in row]
                rows.append(cols)
                for i_col, cell in enumerate(cols):
                    cell: Tuple[str, ...]  # type hint, it's multi-line cell
                    width = max(len(x) for x in cell if not self._is_divider(x))
                    max_width_list[i_col] = max(max_width_list[i_col], width)
                    max_height_list[i_row] = max(max_height_list[i_row], len(cell))

        # * Generate divider
        divider_double = self._divider_double(max_width_list)
        divider_single = self._divider_single(max_width_list)

        def _get_pad_y(h, H):
            assert H >= h
            if self._align_y == "top":
                return 0, H - h
            elif self._align_y == "bottom":
                return H - h, 0
            else:
                return (H - h) // 2, (H - h) - ((H - h) // 2)

        def _pad_width(line, W):
            # * inner cell divider
            if line == self.DIVIDER_SINGLE:
                return "-" * W
            elif line == self.DIVIDER_DOUBLE:
                return "=" * W

            assert W >= len(line)
            pad = W - len(line)
            if len(line) == 0:
                return " " * pad
            # * determine the pad char
            ch = " "
            if line[0] == "-" and line[-1] == "-":
                ch = "-"
            if line[0] == "=" and line[-1] == "=":
                ch = "="

            if self._align_x == "left":
                return line + ch * pad
            elif self._align_x == "right":
                return ch * pad + line
            else:
                return ch * (pad // 2) + line + ch * (pad - pad // 2)

        # * Generate lines
        lines: List[str] = [divider_double]  # start from a divider
        for i_row, row in enumerate(rows):
            # * divider, only when last line is not divider
            if self._is_divider(row):
                if lines[-1] in [divider_double, divider_single]:
                    # change single to double
                    if row == self.DIVIDER_DOUBLE and lines[-1] == divider_single:
                        lines[-1] = divider_double
                else:
                    lines.append(divider_double if row == self.DIVIDER_DOUBLE else divider_single)
                continue
            # * multi-line columns
            H = max_height_list[i_row]
            new_columns = []
            for i_col, cell in enumerate(row):
                W = max_width_list[i_col]
                # * pad height and width according to alignment
                pad_y0, pad_y1 = _get_pad_y(len(cell), H)
                padded_cell = [" " * W for _ in range(pad_y0)]  # pad height
                for line in cell:
                    padded_cell.append(_pad_width(line, W))  # pad width
                padded_cell.extend([" " * W for _ in range(pad_y1)])  # pad height
                new_columns.append(padded_cell)
            for i in range(self._n_columns - len(row)):
                W = max_width_list[i + len(row)]
                padded_cell = [" " * W for _ in range(H)]
                new_columns.append(padded_cell)

            for y in range(len(new_columns[0])):
                line = "|"
                for x in range(len(new_columns)):
                    line += new_columns[x][y] + "|"
                lines.append(line)
        # * Last divider
        if lines[-1] == divider_single or lines[-1] == divider_double:
            lines[-1] = divider_double
        else:
            lines.append(divider_double)

        return "\n".join(lines)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                       Simple Table to LaTex or Markdown                                      * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def to_markdown(self):
        # * We convert all cell into Tuple[str, ...], which can be multi-line.
        rows: List[Optional[List[Tuple[str, ...]]]] = []
        # * Also, get the max width for each column and max height for each row
        max_width_list = [0 for _ in range(self._n_columns)]
        max_height_list = [0 for _ in range(len(self._rows))]
        for i_row, row in enumerate(self._rows):
            if self._is_divider(row) or len(row) == 0:
                pass
            else:
                cols = [str(val) for val in row]
                rows.append(cols)
                for i_col, cell in enumerate(cols):
                    cell: Tuple[str, ...]  # type hint, it's multi-line cell
                    width = len(cell)
                    max_width_list[i_col] = max(max_width_list[i_col], width)
                    max_height_list[i_row] = max(max_height_list[i_row], len(cell))

        def _pad_width(line, W):
            assert W >= len(line)
            pad = W - len(line)
            if len(line) == 0:
                return " " * pad
            # * determine the pad char
            ch = " "
            if line[0] == "-" and line[-1] == "-":
                ch = "-"
            if line[0] == "=" and line[-1] == "=":
                ch = "="

            if self._align_x == "left":
                return line + ch * pad
            elif self._align_x == "right":
                return ch * pad + line
            else:
                return ch * (pad // 2) + line + ch * (pad - pad // 2)

        lines = []
        for i_row, cols in enumerate(rows):
            new_columns = []
            for i_col, cell in enumerate(cols):
                new_columns.append(_pad_width(cell, max_width_list[i_col]))
            lines.append("| " + " | ".join(new_columns) + " |")
            if i_row == 0:

                def _header_line(length):
                    assert length >= 2
                    if self._align_x == "left":
                        return ":" + "-" * (length - 1)
                    elif self._align_x == "right":
                        return "-" * (length - 1) + ":"
                    elif self._align_x == "center":
                        return ":" + "-" * (length - 2) + ":"
                    else:
                        return "-" * length

                new_columns = [_header_line(len(x)) for x in new_columns]
                lines.append("| " + " | ".join(new_columns) + " |")
        return "\n".join(lines)


if __name__ == "__main__":
    table = Table("Name", "Score", alignment=("left", "middle"))
    table.add_row("Brain", 100.0)
    table.add_row("Amaza", 0, ["line0\nline1\nline2\n", "12345"])
    table.add_row("Rosa", 10.5)
    table.divide(table.DIVIDER_SINGLE)
    table.add_row("Rosa2", 20.5, dict(a=10, b=5.5, c=5))

    print(table)
