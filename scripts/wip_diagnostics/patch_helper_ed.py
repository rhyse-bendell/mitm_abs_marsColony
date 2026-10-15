import io
ROOT = r"D:\Post-doc Work\mitm_abs_marsColony" + "\\"
def sub(path, old, new, count=1):
    p = ROOT + path
    raw = open(p, "rb").read().decode("utf-8")
    crlf = "\r\n" in raw
    txt = raw.replace("\r\n", "\n")
    n = txt.count(old)
    if n != count:
        raise SystemExit(f"{path}: expected {count} match(es) of old text, found {n}: {old[:80]!r}")
    txt = txt.replace(old, new)
    if crlf:
        txt = txt.replace("\n", "\r\n")
    open(p, "wb").write(txt.encode("utf-8"))
    print("patched", path)
