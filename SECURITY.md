# Security: how teacher credentials are stored

Short version: **teacher passwords are never stored in a form anyone can read —
not even you.** That is deliberate and it is the safest design. Below is what
lives where, and how to keep it safe.

## Where credentials live

Teacher accounts are stored in:

```
data/teacher_profiles/teachers.json
data/teacher_profiles/vouchers.json
```

These files are **not** tracked in git (see `.gitignore`) and are written with
owner-only permissions (`0600`, in a `0700` directory) so other users on the
host cannot read them.

## Passwords: hashed, not "vaulted"

Each teacher record stores only a **salted PBKDF2-SHA256 hash** of the password
(260,000 iterations), created in `core/security.py`:

```
"password_hash": "pbkdf2_sha256$260000$<salt>$<digest>"
```

The plaintext password is never written to disk or logs. At login it is hashed
again and compared to the stored hash in constant time.

**Why not encrypt passwords into a reversible vault?** Because any vault you can
decrypt, an attacker who steals the key can also decrypt. A one-way hash can't
be reversed by anyone — so even a full copy of `teachers.json` does not reveal
a single password. That is strictly stronger than an encrypted, recoverable
store, which is why we keep passwords hashed.

## Emails

Emails are stored as plaintext inside the same `0600` file, because they are
used as the login identifier and for teacher lookup. They are protected by the
file permissions and by being kept out of git. If you later need them encrypted
at rest as well (e.g. regulatory PII requirements), that is a follow-up: it
requires adding a crypto dependency and managing an encryption key, and it must
preserve exact-match lookups used by login and voucher redemption.

## Keeping it safe — checklist

- [ ] Never commit `data/` (already gitignored — keep it that way).
- [ ] Set a strong, random `SESSION_SECRET` in `.env` (see `.env.example`).
- [ ] Keep `COOKIE_SECURE=1` and serve over HTTPS in production.
- [ ] Never put real API keys in `.env.example` — only in `.env` (gitignored).
- [ ] Back up `data/teacher_profiles/` privately (encrypted backup), since the
      hashes and emails cannot be regenerated if lost.
- [ ] If a secret is ever committed, rotate it — removing it from the file does
      not remove it from git history.
