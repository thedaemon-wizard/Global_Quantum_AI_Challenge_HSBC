# Regulatory sources: official locators and what was checked

`docs/REFERENCES.md` section 5 names six regulatory instruments and says the official locators
are here. They were not, and five of the six had no resolving locator at all — an instrument
named in prose is not a citation a reader can follow.

Every entry below was fetched from the issuing authority or from an official consolidated text
on **2026-08-30**, and the specific provision the project relies on was read rather than
assumed. Where the check changed something, that is stated.

Regulation is cited in this project as a **structural argument** — what a decision has to look
like to be auditable — and never as a threshold selector. Protocol amendment A2 records why:
treating a portfolio-level exemption ceiling as a decline threshold on a fraud-enriched
research benchmark is a category error, and it was one this project made and withdrew.

---

## RG-1 — PSD2 SCA-RTS

**Commission Delegated Regulation (EU) 2018/389** of 27 November 2017, supplementing Directive
(EU) 2015/2366 as regards regulatory technical standards for strong customer authentication and
common and secure open standards of communication.

| | |
|---|---|
| CELEX | `32018R0389` |
| ELI | <https://eur-lex.europa.eu/eli/reg_del/2018/389/oj> |
| Consolidated text used | <https://www.legislation.gov.uk/eur/2018/389> (UK-onshored copy of the same articles; see RG-6) |
| Provisions relied on | Article 18 (transaction risk analysis exemption), Article 19 (fraud rate calculation), Annex (reference fraud rates) |

**Article 19, verified.** The fraud rate is *value*-based and computed over a rolling quarter:
the total value of unauthorised or fraudulent remote transactions divided by the total value of
all remote transactions of the same type, "on a rolling quarterly basis (90 days)". This is why
`src/hsbcfraud/metrics.py` reports a value-weighted quantity and `config.py` sets
`psd2_window_days = 90`.

**The Annex, verified in full.** The table has two columns and reading the wrong one is easy:

| ETV | Remote electronic card-based payments | Remote electronic credit transfers |
|---|---|---|
| EUR 500 | 0.01 % | 0.005 % |
| EUR 250 | 0.06 % | 0.01 % |
| EUR 100 | 0.13 % | 0.015 % |

The project uses the **card-based** column, which is correct for this problem, and
`config.py` reproduces it exactly as `{"100": 0.0013, "250": 0.0006, "500": 0.0001}`.

**Settled 2026-08-30 — Article 21.** `REFERENCES.md` listed Article 21 (monitoring) among the
provisions relied on while no code or document cited it. It was doing no work, so it has been
removed from the entry. The monitoring obligation is real but belongs to a deployment, not to
this study.

## RG-2 — EU AI Act, the fraud-detection exception

**Regulation (EU) 2024/1689**, Annex III point 5(b) and Recital 58.

| | |
|---|---|
| CELEX | `32024R1689` |
| Official Journal | OJ L, 2024/1689, 12 July 2024 |
| Annex III | <https://artificialintelligenceact.eu/annex/3/> |
| Recital 58 | <https://artificialintelligenceact.eu/recital/58/> |

**Annex III 5(b), quoted exactly:** "AI systems intended to be used to evaluate the
creditworthiness of natural persons or establish their credit score, *with the exception of AI
systems used for the purpose of detecting financial fraud*".

**Recital 58 is narrower than the Annex, and the difference matters.** It exempts "AI systems
*provided for by Union law* for the purpose of detecting fraud in the offering of financial
services". The Annex exception is unqualified; the recital's is conditioned on the system being
provided for by Union law. This project makes **no high-risk compliance claim either way**, so
nothing here turns on which reading prevails — but a submission that cited the exception as
settled would be overstating it, and this file records why it is not cited that way.

## RG-3 — Deferral of stand-alone Annex III obligations

**Regulation (EU) 2026/1744**, the Digital Omnibus on AI. Adopted 8 July 2026, published in the
Official Journal 24 July 2026. Defers stand-alone Annex III high-risk obligations to
2 December 2027.

Verified on EUR-Lex. Recorded because it changes *when* the RG-2 question would arise, not
whether this project answers it.

## RG-4 — US model risk management

**SR 26-2**, *Revised Guidance on Model Risk Management*, issued 17 April 2026 jointly by the
Board of Governors of the Federal Reserve System, the OCC and the FDIC. Supersedes SR 11-7 and
SR 21-8.

| | |
|---|---|
| Issuer | Federal Reserve, OCC, FDIC |
| Locator | <https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm> |

Verified at the issuer on 2026-09-05: the number, the title, the 17 April 2026 date, all three
agencies, and both superseded letters -- SR 11-7 (4 April 2011) and SR 21-8 (9 April 2021) --
match the Federal Reserve's own page.

**Scope, which the letter states and this entry did not.** It applies primarily to banking
organizations above **30 billion dollars in assets**, and describes "a risk-based approach to
model risk management that is tailored to a banking organization's model risk profile". The
sponsor is far above that threshold, so applicability is not in question -- but the entry
asserted relevance without recording the test, and a reviewer checking the source finds the
threshold before finding anything else. Recorded because an applicability claim that omits the
applicability criterion is the shape of [D-129](decisions.md)'s error.

Relied on for the structure of what a model-risk function asks: validation, ongoing monitoring,
outcome analysis, and benchmarking against other models. That structure is why the submission
reports a certificate and a held-out validation rather than a leaderboard score, and why the
feature attribution in `attribution.csv` exists at all.

## RG-5 — UK model risk management

**PRA Supervisory Statement SS1/23**, *Model risk management principles for banks*.

| | |
|---|---|
| Issuer | Prudential Regulation Authority, Bank of England |
| Published | 17 May 2023; **current version 23 April 2026** |
| Effective | 17 May 2024; current version effective 23 April 2026 |
| Locator | <https://www.bankofengland.co.uk/prudential-regulation/publication/2023/may/model-risk-management-principles-for-banks-ss> |

Sets out five principles for model risk management.

**The version was wrong, and the fetch date is what makes that awkward.** Re-checked at the
Bank of England on 2026-09-06: the page lists an April 2026 edition as current and the May 2023
edition as *past*, the current one "Published 23 April 2026. Effective from 23 April 2026",
following **LIAF01/26 -- Low Impact Amendments Finalisation April 2026**. This entry recorded
only the 2023 dates, and recorded them as verified on 2026-08-30 -- four months after the
amendment was live. Verifying a date on the issuer's page is not the same as checking which
version that page is serving.

What this project takes from SS1/23 is the scope argument in paragraphs 1.2 to 1.4, and the
amendment is a low-impact finalisation rather than a rewrite.

**Settled 2026-09-06, and the marker is closed.** This entry carried a **Needs confirmation**
saying the paragraphs relied on had been read only in the 2023 text and that no claim was made
about what LIAF01/26 altered in them. Both versions have now been fetched from the Bank of
England and the three paragraphs compared word by word:

| Paragraph | 2023 text vs April 2026 text |
|---|---|
| 1.2 -- scope limited to firms with internal model approval | **identical**, 677 characters both |
| 1.3 -- all models, in-house or vendor, including financial reporting | **identical**, 738 characters both |
| 1.4 -- five principles "across all model and risk types" | **identical** body, to "in its own right" |

The only difference the comparison surfaced in 1.4 was a footnote that the 2026 PDF's text layer
places inline and the 2023 one does not -- a typesetting artefact, not an amendment. So the scope
argument this project makes stands on the **current** version, and the paragraph numbers it cites
still point at the text it quotes.

Sources: `liaf0126app5.pdf` (current, April 2026) and `ss123.pdf` (past, May 2023), both from the
publication page above. See [D-140](decisions.md) and [D-151](decisions.md).

**Settled 2026-08-30 — the applicability sentence.** `REFERENCES.md` previously stated SS1/23
was "confirmed by the PRA to apply to fraud models". No such confirmation exists at the issuing
authority — the word "fraud" does not occur in SS1/23 — so the claim was withdrawn. What
supports the citation instead is the statement's own scope: paragraph 1.2 limits it to firms
with internal model approval, and within that scope paragraphs 1.3--1.4 cover all model and risk
types. `REFERENCES.md` now says exactly that, so this item is closed.

## RG-6 — UK onshored SCA-RTS

**Corrected during this check.** `REFERENCES.md` attributed the onshoring to **PS21/19**. It
did not do that.

| Policy statement | What it actually is |
|---|---|
| **PS19/26** | *Brexit — Regulatory Technical Standards for Strong Customer Authentication and Common and Secure Open Standards of Communication.* This is the instrument that **onshored** the SCA-RTS into UK law. <https://www.fca.org.uk/publications/policy-statements/ps19-26-brexit-regulatory-technical-standards-strong-customer-authentication> |
| **PS21/19** | *Changes to the SCA-RTS and to guidance in the Approach Document and the Perimeter Guidance Manual*, 29 November 2021, in force 26 March 2022. This **amends** the onshored SCA-RTS — a new Article 10A exemption and Article 36(6) consent reconfirmation, both open-banking measures. <https://www.fca.org.uk/publications/policy-statements/ps21-19-changes-sca-rts-and-guidance-approach-document-and-perimeter-guidance-manual> |

Neither changes the Annex reference fraud rates or the Article 18/19 structure that RG-1
supplies, so no figure in this project moves. The citation was simply pointing at the wrong
document, and the entry now points at both with their roles distinguished.

**Pending revocation.** `legislation.gov.uk` marks the onshored regulation "Regulation
revoked by 2023 c. 29 Sch. 1 Pt. 3" under *changes yet to be applied* — the Financial Services
and Markets Act 2023 legislates the revocation of onshored financial services law, to be
replaced under the Smarter Regulatory Framework, but the revocation of the SCA-RTS had not been
commenced as of this check. The UK position is therefore **in transition**, and any statement
in this submission about UK applicability is written in the present tense deliberately.

---

## How these were checked

Regulatory text resists automated summarising, and one attempt during this check read the
*credit transfers* column of the RG-1 Annex and reported it as the card-based rates — which
would have made the repository look wrong when it is right. Locators here were confirmed by
extracting the table structure from the source markup rather than by asking a summariser what
it said. Where a claim could not be confirmed at the issuing authority, it is marked Needs confirmation
above rather than repeated.
