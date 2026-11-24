    .section .rodata.koko_blob, "a"
    .balign 4
    .global koko_blob
koko_blob:
    .incbin "build/koko.bin"
