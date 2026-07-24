# Opening books (Polyglot .bin)

- `komodo.bin` — the free opening book distributed with the Komodo chess engine, obtained from
  [gmcheems-org/free-opening-books](https://github.com/gmcheems-org/free-opening-books) (578,126 entries).

Enable in Zenith with:

    setoption name BookFile value books/komodo.bin
    setoption name OwnBook value true

`OwnBook` defaults to **false**: engine testing (SPRT, bench) must stay bookless, and this repo's own
measurement history shows why — an opponent's default-on book once contaminated days of gap measurements.
