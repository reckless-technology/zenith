# Opening books (Polyglot `.bin`)

Zenith supports the standard **Polyglot** `.bin` opening-book format, and ships its own book, built entirely from
Zenith's self-play:

- `zenith-book-r5.bin` - the shipped book, embedded in every binary: r4 with 1.d4 added at the root at weight 42
  (`tools/add_book_moves.py`), so Zenith opens 1.e4, 1.c4 and 1.d4, the last about a fifth of the time.
- `zenith-book-r4.bin` - `tools/build_book.py`'s coverage book: replies to plausible moves Zenith would not choose
  itself.
- `zenith-book-r3.bin` - the earlier book, without that coverage.

```
setoption name OwnBook value true          # the embedded book; default false - testing always runs bookless
setoption name BookFile value path/to/your-book.bin   # or a book of your own instead
```

Any Polyglot `.bin` works as a BookFile; verify a third-party book's redistribution terms before bundling it.
`./build/zenith bookcheck` validates Zenith's Polyglot key computation against the 9 official spec vectors, and
`./build/zenith embookcheck` the embedded book. `tools/show_book.py` prints a book as SAN lines.
