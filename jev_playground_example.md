# Jev playground example

Paste each block into its own field in the TypeSafe playground. The questions box takes only the questions
object. If it reports invalid JSON, paste the single-line version (the editor can mangle multi-line pastes),
and if the full set is rejected, paste one question at a time.

## Model

```
jev-latest
```

## State (plain text, not JSON)

Good translation:

```
English source:
King (L):

moves in any direction one step.

def king_moves(x,y):


Hindi translation:
राजा (L):

एक कदम किसी भी दिशा में गति करता है।

def king_moves(x,y):
```

## Questions (single line)

```
{"correct":{"type":"noul","instructions":"Is the Hindi a faithful, complete and fluent translation of the English? Math equations, formulas, numbers, tables, code and variable names must stay exactly as in the English, so a verbatim copy of such content is correct and must not be penalized. Only the surrounding English prose should be translated into Hindi. Consider missing content, mistranslation, garbling, repetition, prose left in English and math or code that was altered or transliterated.","criteria":{"true":"Prose is faithfully translated into Hindi, and math, numbers and code are kept exactly unchanged","false":"Prose is missing, wrong, garbled, repeated or left in English, or math or code was altered"}},"complete":{"type":"noul","instructions":"Is every part of the English content present in the Hindi, with nothing dropped or cut off?","criteria":{"true":"All content is present and the Hindi ends where the English ends","false":"Part of the content is missing or the Hindi is cut off"}},"prose_translated":{"type":"noul","instructions":"Is all of the English prose translated into Hindi, apart from math, code and technical terms that may stay in English?","criteria":{"true":"All prose is written in Hindi in Devanagari script","false":"Some prose is left in English or written in romanized Hindi"}},"math_preserved":{"type":"noul","instructions":"Are all equations, numbers, code and variable names in the Hindi identical to the English?","criteria":{"true":"Every equation, number and piece of code is unchanged","false":"Some math, number or code was changed, transliterated or dropped"}},"fluent":{"type":"noul","instructions":"Is the Hindi free of repetition and garbled text, and does it read naturally?","criteria":{"true":"Natural Hindi with no repeated loops and no corrupted characters","false":"Contains repeated phrases, corrupted characters or unreadable text"}},"issue":{"type":"choice","instructions":"What is the main problem with the Hindi translation, if any?","criteria":{"none":"No real problem","truncated":"Content is cut off or missing","repeated":"Phrases repeat in a loop","untranslated":"Prose is left in English or in romanized Hindi","math_altered":"Math, numbers or code were changed","garbled":"Corrupted or unreadable text","wrong_content":"The Hindi does not match the English"}}}
```

The `correct` question is the overall verdict. `complete`, `prose_translated`, `math_preserved` and `fluent` say
which aspect is wrong, and `issue` picks the main problem. Results measured with the repo client on these states:

| Hindi text | correct | complete | prose_translated | math_preserved | fluent | issue |
|---|---|---|---|---|---|---|
| good translation | 0.93 | 0.78 | 0.81 | 0.92 | 0.93 | none |
| `राजा राजा राजा राजा राजा` | 0.01 | 0.02 | 0.20 | 0.10 | 0.01 | repeated |
| English copied unchanged | 0.09 | 0.89 | 0.07 | 0.96 | 0.50 | untranslated |

## Bad states to compare

Repeated words (same model and questions):

```
English source:
King (L):

moves in any direction one step.

def king_moves(x,y):


Hindi translation:
राजा राजा राजा राजा राजा
```

English left untranslated:

```
English source:
King (L):

moves in any direction one step.

def king_moves(x,y):


Hindi translation:
King (L):

moves in any direction one step.

def king_moves(x,y):

```

## Same request over HTTP

```json
{
  "model": "jev-latest",
  "state": "English source:\nKing (L):\n\nmoves in any direction one step.\n\ndef king_moves(x,y):\n\n\nHindi translation:\nराजा (L):\n\nएक कदम किसी भी दिशा में गति करता है।\n\ndef king_moves(x,y):",
  "questions": {
    "correct": {
      "type": "noul",
      "instructions": "Is the Hindi a faithful, complete and fluent translation of the English? Math equations, formulas, numbers, tables, code and variable names must stay exactly as in the English, so a verbatim copy of such content is correct and must not be penalized. Only the surrounding English prose should be translated into Hindi. Consider missing content, mistranslation, garbling, repetition, prose left in English and math or code that was altered or transliterated.",
      "criteria": {
        "true": "Prose is faithfully translated into Hindi, and math, numbers and code are kept exactly unchanged",
        "false": "Prose is missing, wrong, garbled, repeated or left in English, or math or code was altered"
      }
    },
    "complete": {
      "type": "noul",
      "instructions": "Is every part of the English content present in the Hindi, with nothing dropped or cut off?",
      "criteria": {
        "true": "All content is present and the Hindi ends where the English ends",
        "false": "Part of the content is missing or the Hindi is cut off"
      }
    },
    "prose_translated": {
      "type": "noul",
      "instructions": "Is all of the English prose translated into Hindi, apart from math, code and technical terms that may stay in English?",
      "criteria": {
        "true": "All prose is written in Hindi in Devanagari script",
        "false": "Some prose is left in English or written in romanized Hindi"
      }
    },
    "math_preserved": {
      "type": "noul",
      "instructions": "Are all equations, numbers, code and variable names in the Hindi identical to the English?",
      "criteria": {
        "true": "Every equation, number and piece of code is unchanged",
        "false": "Some math, number or code was changed, transliterated or dropped"
      }
    },
    "fluent": {
      "type": "noul",
      "instructions": "Is the Hindi free of repetition and garbled text, and does it read naturally?",
      "criteria": {
        "true": "Natural Hindi with no repeated loops and no corrupted characters",
        "false": "Contains repeated phrases, corrupted characters or unreadable text"
      }
    },
    "issue": {
      "type": "choice",
      "instructions": "What is the main problem with the Hindi translation, if any?",
      "criteria": {
        "none": "No real problem",
        "truncated": "Content is cut off or missing",
        "repeated": "Phrases repeat in a loop",
        "untranslated": "Prose is left in English or in romanized Hindi",
        "math_altered": "Math, numbers or code were changed",
        "garbled": "Corrupted or unreadable text",
        "wrong_content": "The Hindi does not match the English"
      }
    }
  }
}
```

```bash
curl https://api.typesafe.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  -d @request.json
```

Or with the repo client (`rl/typesafe_judge.py`, key in `.env`):

```python
from rl.typesafe_judge import TypesafeJudge
print(TypesafeJudge().judge(en, hi))   # JudgeResult with p_correct, per-question scores and the main issue
```
