# Opening books (Polyglot `.bin`)

Zenith supports the standard **Polyglot** `.bin` opening-book format. No book ships in this repo — the
format's licensing across the community is murky, so bring your own:

```
setoption name BookFile value path/to/your-book.bin
setoption name OwnBook value true          # default false — testing always runs bookless
```

Any Polyglot `.bin` works. Freely-available collections exist (search "polyglot opening book"); verify a
book's own redistribution terms before bundling it. `./zenith bookcheck` validates Zenith's Polyglot key
computation against the 9 official spec vectors, independent of any book.
