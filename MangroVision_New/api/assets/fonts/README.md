# Report fonts

The PDF renderer embeds static regular (400) and bold (700) instances of
[Inter](https://github.com/google/fonts/tree/main/ofl/inter), at optical size 14.
The font's SIL Open Font License is included in `OFL.txt`.

The internal names are `MangroVisionInter-Regular` and `MangroVisionInter-Bold`
so PDF libraries distinguish the static faces. No network font request is
needed when saving a report.
