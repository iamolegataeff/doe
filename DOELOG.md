# DOELOG

The running engineering log of DOE — Democracy of Experts. Every fix, every
verified change, every bug-class closed — dated, with commit and proof. `README.md`
is the spec and the manifesto; **this is the work**.

Convention: small fixes (bug fixes, hardening, single-op work, docstring touch-ups)
are recorded **here**. Large changes (a new backend, a new physics term, an
architecture shift) get a section in the README too. When in doubt: it goes here
first. Entries are **de-personalized and technical** — what changed, where
(`file:line`), and the reproducible proof. No signatures in the log.

Newest entries on top.

---

## 2026-07-20 — integer-overflow hardening: alloc/copy sizes widened to `size_t`

CodeQL (default setup, threat model `remote`) flagged 16
`cpp/integer-multiplication-cast-to-long` alerts in `doe.c`: `int * int` products
used as allocation or copy sizes that overflow in `int` before the implicit widen to
`size_t`. A crafted or oversized host profile could wrap the product to a small value
and under-allocate a LoRA / KV / RoPE buffer, then write past its end.

Each flagged size expression now casts its leading operand to `size_t`, so the product
computes in the wide type:

- **LoRA experts** — `init_lora_expert` (`doe.c:1749`), `mycelium_save` (`:2662`),
  `mycelium_load` (`:2722`, `:2728`): `dim*rank` / `rank*dim` in `calloc`, `fwrite`, and
  `fread` — the readback comparison `(size_t)(dim*rank)` was itself an int product cast
  after the fact and is now `(size_t)dim * rank`.
- **inference state** — `alloc_infer` (`doe.c:2776`–`2804`): `host_heads*host_head_dim`,
  `host_heads*max_seq`, `host_n_layers*max_seq*kd`, `max_seq*half` in `calloc`.
- **KV reset** — `chat` (`doe.c:3639`): `host_n_layers*max_seq*kd*4` in `memset` — now
  matches the already-cast `kv_bytes` and the sibling reset at `:4050`.

Numerically inert for in-range sizes; C cast precedence makes `(size_t)A*B` compute as
`((size_t)A)*B`. Proof (neo): `make test` → **113/113**; `cc -O2 -Wall -Wextra -c doe.c`
clean apart from the one pre-existing `unused 'hs'` warning. Canon fix — the
`yent-inference` vendored copy was already patched by Codex; this closes the source.
