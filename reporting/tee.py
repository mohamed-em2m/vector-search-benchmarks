import textwrap

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
            except UnicodeEncodeError:
                f.write(data.encode('ascii', 'replace').decode('ascii'))
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.files[0].fileno()

    @property
    def encoding(self):
        return getattr(self.files[0], "encoding", "utf-8")

def snippet(text: str, width=70) -> str:
    text = " ".join(text.split())
    return textwrap.shorten(text, width=width, placeholder="…")

def sep(char="─", width=90):
    print(char * width)

def header(title: str):
    sep("═")
    print(f"  {title}")
    sep("═")

def section(title: str):
    print()
    sep()
    print(f"  {title}")
    sep()
