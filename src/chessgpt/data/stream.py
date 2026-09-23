import io
from collections.abc import Generator
import zstandard


def stream_blocks(path: str) -> Generator[str]:
    decompressor = zstandard.ZstdDecompressor()
    with open(path, "rb") as file, decompressor.stream_reader(file) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        headers = {}
        moves_lines = []
        in_moves = False

        for line in text:
            line = line.strip()
            if not line:
                continue

            if line.startswith("["):
                if in_moves:
                    yield headers, " ".join(moves_lines)
                    headers, moves_lines, in_moves = {}, [], False

                key, _, rest = line[1:-1].partition(" ")
                headers[key] = rest.strip('"')
            else:
                in_moves = True
                moves_lines.append(line)

        if headers:
            yield headers, " ".join(moves_lines)
