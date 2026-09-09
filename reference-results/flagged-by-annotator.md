# Elements the annotator flagged — not found, or ambiguous

While adjudicating the two model labellings into `data/dataset-v1-human.jsonl`, the annotator saw each
element's three descriptions and the labeller's box, and could flag instead of drawing. **55** elements were
flagged *not found* — the description names something that is not on the screen — and **37** *ambiguous* —
the description could fit more than one thing, or nothing decisively. Both kinds were left out of the ground
truth, so no benchmark figure includes them.

The not-found set is the one place this corpus shows a labeller inventing an element (46 of the 55 carry
GPT-5.6's descriptions, 9 Opus's; 16 are the same pattern, a guest-count "increase" stepper (adults, children, infants, pets) on one app).
It is also the natural seed for measuring whether a model will answer *not found* when asked about something
absent — see the README under [When it is wrong, what did it do?](../README.md#when-it-is-wrong-what-did-it-do).

Each line names the CVAT task and frame the annotator worked on, the element id as it appears in the
labelling (`gpt-`/`opus-` prefix = whose description), and the `name` description. The links open the frame
in CVAT and need a CVAT login; the screenshot itself is `data/images/<screenshot_id>.png`, and the two
labellings' boxes for it are in `data/dataset-v1-gpt-5.6.jsonl` and `data/dataset-v1.jsonl`.

## not_found (55)

- [boxes / task 2535202 / frame 108](https://app.cvat.ai/tasks/2535202/jobs/4394360?frame=108) — gpt-e12: “the infants increase button”
- [boxes / task 2535202 / frame 109](https://app.cvat.ai/tasks/2535202/jobs/4394360?frame=109) — gpt-e10: “the children increase button”
- [boxes / task 2535202 / frame 175](https://app.cvat.ai/tasks/2535202/jobs/4394360?frame=175) — gpt-e8: “the increase adults button”
- [boxes / task 2535202 / frame 262](https://app.cvat.ai/tasks/2535202/jobs/4394361?frame=262) — gpt-e8: “the adults increase button”
- [boxes / task 2535202 / frame 275](https://app.cvat.ai/tasks/2535202/jobs/4394361?frame=275) — gpt-e14: “the pets increase button”
- [boxes / task 2535202 / frame 282](https://app.cvat.ai/tasks/2535202/jobs/4394361?frame=282) — gpt-e12: “the infant increase control”
- [boxes / task 2535202 / frame 415](https://app.cvat.ai/tasks/2535202/jobs/4394363?frame=415) — gpt-e12: “the infants increase button”
- [boxes / task 2535202 / frame 427](https://app.cvat.ai/tasks/2535202/jobs/4394363?frame=427) — gpt-e14: “the pet increase control”
- [boxes / task 2535202 / frame 440](https://app.cvat.ai/tasks/2535202/jobs/4394363?frame=440) — gpt-e14: “the increase pets button”
- [boxes / task 2535207 / frame 77](https://app.cvat.ai/tasks/2535207/jobs/4394369?frame=77) — gpt-e10: “the child increase control”
- [boxes / task 2535207 / frame 293](https://app.cvat.ai/tasks/2535207/jobs/4394371?frame=293) — gpt-e8: “the adult increase control”
- [boxes / task 2535207 / frame 294](https://app.cvat.ai/tasks/2535207/jobs/4394371?frame=294) — gpt-e10: “the children increase control”
- [boxes / task 2535207 / frame 295](https://app.cvat.ai/tasks/2535207/jobs/4394371?frame=295) — gpt-e8: “the adults increase control”
- [boxes / task 2535211 / frame 268](https://app.cvat.ai/tasks/2535211/jobs/4394379?frame=268) — gpt-e10: “the increase children button”
- [boxes / task 2535211 / frame 271](https://app.cvat.ai/tasks/2535211/jobs/4394379?frame=271) — gpt-e12: “the increase infants button”
- [boxes / task 2535211 / frame 338](https://app.cvat.ai/tasks/2535211/jobs/4394380?frame=338) — gpt-e12: “the infants increase control”
- [boxes / task 2535211 / frame 369](https://app.cvat.ai/tasks/2535211/jobs/4394380?frame=369) — gpt-e11: “the add dates button”
- [boxes / task 2535220 / frame 86](https://app.cvat.ai/tasks/2535220/jobs/4394388?frame=86) — gpt-e5: “the Monday 11 forecast”
- [boxes / task 2535220 / frame 187](https://app.cvat.ai/tasks/2535220/jobs/4394389?frame=187) — gpt-e1: “the Thursday 7 forecast”
- [boxes / task 2535220 / frame 235](https://app.cvat.ai/tasks/2535220/jobs/4394390?frame=235) — gpt-e22: “the Matt Goodwin preview”
- [boxes / task 2535220 / frame 365](https://app.cvat.ai/tasks/2535220/jobs/4394391?frame=365) — gpt-e3: “the English language tab”
- [boxes / task 2535220 / frame 378](https://app.cvat.ai/tasks/2535220/jobs/4394391?frame=378) — gpt-e4: “the Sunday 10 forecast”
- [existence / task 2535232 / frame 107](https://app.cvat.ai/tasks/2535232/jobs/4394406?frame=107) — gpt-e19: “the February spreadsheet options”
- [existence / task 2535232 / frame 454](https://app.cvat.ai/tasks/2535232/jobs/4394409?frame=454) — gpt-e4: “the keyboard clipboard button”
- [existence / task 2535253 / frame 86](https://app.cvat.ai/tasks/2535253/jobs/4394432?frame=86) — gpt-e9: “the bright condo listing”
- [existence / task 2535253 / frame 87](https://app.cvat.ai/tasks/2535253/jobs/4394432?frame=87) — gpt-e10: “the condo listing favorite button”
- [existence / task 2535253 / frame 383](https://app.cvat.ai/tasks/2535253/jobs/4394435?frame=383) — gpt-e14: “the special character key”
- [existence / task 2535253 / frame 394](https://app.cvat.ai/tasks/2535253/jobs/4394435?frame=394) — gpt-e15: “the Full size filter”
- [existence / task 2535253 / frame 464](https://app.cvat.ai/tasks/2535253/jobs/4394436?frame=464) — gpt-e10: “the Al Jazeera follow star”
- [existence / task 2535253 / frame 466](https://app.cvat.ai/tasks/2535253/jobs/4394436?frame=466) — gpt-e18: “the Journal follow star”
- [existence / task 2535253 / frame 493](https://app.cvat.ai/tasks/2535253/jobs/4394436?frame=493) — gpt-e19: “the mural illustration”
- [existence / task 2535253 / frame 497](https://app.cvat.ai/tasks/2535253/jobs/4394436?frame=497) — opus-e2: “the close player button”
- [existence / task 2535266 / frame 14](https://app.cvat.ai/tasks/2535266/jobs/4394444?frame=14) — gpt-e7: “the keyboard clipboard shortcut”
- [existence / task 2535266 / frame 370](https://app.cvat.ai/tasks/2535266/jobs/4394447?frame=370) — opus-e10: “the profile tab”
- [existence / task 2535266 / frame 375](https://app.cvat.ai/tasks/2535266/jobs/4394447?frame=375) — gpt-e9: “the profile tab”
- [existence / task 2535266 / frame 376](https://app.cvat.ai/tasks/2535266/jobs/4394447?frame=376) — opus-e3: “the top home tab”
- [existence / task 2535266 / frame 377](https://app.cvat.ai/tasks/2535266/jobs/4394447?frame=377) — opus-e4: “the top library tab”
- [existence / task 2535266 / frame 378](https://app.cvat.ai/tasks/2535266/jobs/4394447?frame=378) — opus-e5: “the top more tab”
- [existence / task 2535266 / frame 479](https://app.cvat.ai/tasks/2535266/jobs/4394448?frame=479) — gpt-e9: “the autumn pavement pin”
- [existence / task 2535266 / frame 480](https://app.cvat.ai/tasks/2535266/jobs/4394448?frame=480) — gpt-e10: “the lower image pin”
- [existence / task 2535283 / frame 88](https://app.cvat.ai/tasks/2535283/jobs/4394463?frame=88) — gpt-e13: “the delete contact option”
- [existence / task 2535283 / frame 122](https://app.cvat.ai/tasks/2535283/jobs/4394464?frame=122) — opus-e1: “the top skip button”
- [existence / task 2535283 / frame 123](https://app.cvat.ai/tasks/2535283/jobs/4394464?frame=123) — opus-e2: “the top next button”
- [existence / task 2535283 / frame 324](https://app.cvat.ai/tasks/2535283/jobs/4394466?frame=324) — gpt-e4: “the terms of service link”
- [existence / task 2535283 / frame 325](https://app.cvat.ai/tasks/2535283/jobs/4394466?frame=325) — opus-e3: “the profile avatar”
- [existence / task 2535283 / frame 391](https://app.cvat.ai/tasks/2535283/jobs/4394466?frame=391) — opus-e3: “the chat icon”
- [consensus / task 2542101 / frame 273](https://app.cvat.ai/tasks/2542101/jobs/4403355?frame=273) — gpt-e7: “the keyboard clipboard control”
- [consensus / task 2542101 / frame 276](https://app.cvat.ai/tasks/2542101/jobs/4403355?frame=276) — gpt-e10: “the keyboard clipboard control”
- [consensus / task 2542109 / frame 210](https://app.cvat.ai/tasks/2542109/jobs/4403366?frame=210) — gpt-e10: “the profile tab”
- [consensus / task 2542134 / frame 209](https://app.cvat.ai/tasks/2542134/jobs/4403405?frame=209) — gpt-e7: “the people results tab”
- [consensus / task 2542137 / frame 31](https://app.cvat.ai/tasks/2542137/jobs/4403411?frame=31) — gpt-e13: “the Testaments included item”
- [consensus / task 2542148 / frame 319](https://app.cvat.ai/tasks/2542148/jobs/4403431?frame=319) — gpt-e2: “the Friday 8 forecast”
- [consensus / task 2542159 / frame 158](https://app.cvat.ai/tasks/2542159/jobs/4403451?frame=158) — gpt-e3: “the experienced learner option”
- [consensus / task 2542166 / frame 21](https://app.cvat.ai/tasks/2542166/jobs/4403459?frame=21) — gpt-e8: “the new featured story”
- [consensus / task 2542166 / frame 64](https://app.cvat.ai/tasks/2542166/jobs/4403459?frame=64) — gpt-e8: “the new featured story”

## ambiguous (37)

- [boxes / task 2535202 / frame 220](https://app.cvat.ai/tasks/2535202/jobs/4394361?frame=220) — gpt-e5: “the Nest Terms link”
- [boxes / task 2535202 / frame 446](https://app.cvat.ai/tasks/2535202/jobs/4394363?frame=446) — gpt-e4: “the learn more link”
- [boxes / task 2535211 / frame 102](https://app.cvat.ai/tasks/2535211/jobs/4394378?frame=102) — gpt-e17: “the July 9 date”
- [boxes / task 2535220 / frame 408](https://app.cvat.ai/tasks/2535220/jobs/4394392?frame=408) — gpt-e8: “the Promotions category row”
- [boxes / task 2535220 / frame 434](https://app.cvat.ai/tasks/2535220/jobs/4394392?frame=434) — gpt-e11: “the add to group option”
- [boxes / task 2535220 / frame 443](https://app.cvat.ai/tasks/2535220/jobs/4394392?frame=443) — gpt-e9: “the Russian language option”
- [existence / task 2535232 / frame 57](https://app.cvat.ai/tasks/2535232/jobs/4394405?frame=57) — gpt-e7: “the publications results tab”
- [existence / task 2535232 / frame 117](https://app.cvat.ai/tasks/2535232/jobs/4394406?frame=117) — opus-e6: “the user agreement link”
- [existence / task 2535232 / frame 137](https://app.cvat.ai/tasks/2535232/jobs/4394406?frame=137) — gpt-e7: “the publications results tab”
- [existence / task 2535232 / frame 340](https://app.cvat.ai/tasks/2535232/jobs/4394408?frame=340) — gpt-e15: “the Arabic download”
- [existence / task 2535253 / frame 32](https://app.cvat.ai/tasks/2535253/jobs/4394432?frame=32) — gpt-e7: “the transit stop markers”
- [existence / task 2535253 / frame 88](https://app.cvat.ai/tasks/2535253/jobs/4394432?frame=88) — gpt-e11: “the living room listing”
- [existence / task 2535253 / frame 89](https://app.cvat.ai/tasks/2535253/jobs/4394432?frame=89) — gpt-e12: “the living room favorite button”
- [existence / task 2535253 / frame 415](https://app.cvat.ai/tasks/2535253/jobs/4394436?frame=415) — gpt-e15: “the profile tab”
- [existence / task 2535266 / frame 305](https://app.cvat.ai/tasks/2535266/jobs/4394447?frame=305) — gpt-e10: “the Teams tab”
- [existence / task 2535266 / frame 324](https://app.cvat.ai/tasks/2535266/jobs/4394447?frame=324) — opus-e7: “the fill arrow for testosterone”
- [existence / task 2535266 / frame 345](https://app.cvat.ai/tasks/2535266/jobs/4394447?frame=345) — opus-e4: “the right cluster marker two”
- [existence / task 2535266 / frame 363](https://app.cvat.ai/tasks/2535266/jobs/4394447?frame=363) — gpt-e14: “the Arabic download button”
- [existence / task 2535283 / frame 52](https://app.cvat.ai/tasks/2535283/jobs/4394463?frame=52) — opus-e14: “the one forty one marker”
- [consensus / task 2542101 / frame 475](https://app.cvat.ai/tasks/2542101/jobs/4403357?frame=475) — gpt-e6: “the eastern two marker”
- [consensus / task 2542126 / frame 346](https://app.cvat.ai/tasks/2542126/jobs/4403391?frame=346) — gpt-e6: “the terms and conditions link”
- [consensus / task 2542126 / frame 382](https://app.cvat.ai/tasks/2542126/jobs/4403391?frame=382) — gpt-e6: “the terms and conditions link”
- [consensus / task 2542131 / frame 167](https://app.cvat.ai/tasks/2542131/jobs/4403396?frame=167) — gpt-e11: “the privacy policy link”
- [consensus / task 2542131 / frame 332](https://app.cvat.ai/tasks/2542131/jobs/4403398?frame=332) — gpt-e9: “the terms conditions link”
- [consensus / task 2542134 / frame 183](https://app.cvat.ai/tasks/2542134/jobs/4403404?frame=183) — gpt-e12: “the privacy policy link”
- [consensus / task 2542134 / frame 247](https://app.cvat.ai/tasks/2542134/jobs/4403405?frame=247) — gpt-e3: “the Manage settings link”
- [consensus / task 2542134 / frame 395](https://app.cvat.ai/tasks/2542134/jobs/4403406?frame=395) — gpt-e3: “the Manage settings link”
- [consensus / task 2542137 / frame 118](https://app.cvat.ai/tasks/2542137/jobs/4403412?frame=118) — gpt-e7: “the terms of use link”
- [consensus / task 2542137 / frame 447](https://app.cvat.ai/tasks/2542137/jobs/4403415?frame=447) — gpt-e1: “the terms of service link”
- [consensus / task 2542137 / frame 478](https://app.cvat.ai/tasks/2542137/jobs/4403415?frame=478) — gpt-e4: “the terms of service link”
- [consensus / task 2542140 / frame 231](https://app.cvat.ai/tasks/2542140/jobs/4403418?frame=231) — gpt-e4: “the Google Account link”
- [consensus / task 2542142 / frame 255](https://app.cvat.ai/tasks/2542142/jobs/4403423?frame=255) — gpt-e1: “the terms of service link”
- [consensus / task 2542142 / frame 498](https://app.cvat.ai/tasks/2542142/jobs/4403425?frame=498) — gpt-e1: “the choices information link”
- [consensus / task 2542148 / frame 495](https://app.cvat.ai/tasks/2542148/jobs/4403432?frame=495) — gpt-e18: “the Hong Kong Chinese reorder handle”
- [consensus / task 2542155 / frame 148](https://app.cvat.ai/tasks/2542155/jobs/4403442?frame=148) — gpt-e6: “the saved contacts link”
- [consensus / task 2542155 / frame 294](https://app.cvat.ai/tasks/2542155/jobs/4403443?frame=294) — gpt-e6: “the saved contacts link”
- [consensus / task 2542159 / frame 256](https://app.cvat.ai/tasks/2542159/jobs/4403452?frame=256) — gpt-e13: “the $121 home price”
