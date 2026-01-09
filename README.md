# Imitation Game

Turing test party game - člověk vs AI soudci. Cílem hráče je přesvědčit AI modely, že je člověk. Kdo vyhraje, dostane pivo.

## Kontext a motivace

Vzniklo z debaty s Ondřejem o tom, jestli má smysl dělat "obrácený Turing test" kde AI soudí lidi. Jeho argument byl o **epistemickém uzávěru** - že AI modely mají uniformní představu lidství a budou rozpoznávat své vlastní quirky jako "lidské".

**Protiargumenty:**
1. Moderní modely NEJSOU homogenní (GPT, Claude, Gemini, Llama mají odlišné "osobnosti")
2. Base modely / prefill mód nemají "asistentskou personu" - úplně jiná epistemologie
3. Turing test nikdy nebyl rigorózní test vědomí, jen schopnost oklamat

**Klíčový insight:** Pokud lidi konzistentně prohrávají proti heterogenní skupině AI soudců, to JE ten finding - ne bug, ale feature.

## Architektura

```
imitgame/
├── __init__.py
├── cli.py              # CLI rozhraní (play, demo, test)
├── game.py             # Core game logic (ImitationGame class)
└── providers/
    ├── __init__.py
    ├── base.py         # Abstract Provider, Message, HumanProvider
    ├── openrouter.py   # OpenRouter provider (standard chat API)
    └── gemini_prefill.py  # Gemini v prefill módu (continuation-style)
```

### Provider interface

```python
class Provider(ABC):
    @property
    def name(self) -> str: ...
    
    def respond(self, messages: list[Message], actor_id: str) -> str:
        """Generuj odpověď v konverzaci."""
        ...
    
    def respond_vote(self, messages: list[Message], actor_id: str) -> str:
        """Generuj hlasování. Override pro providery co potřebují jiné chování."""
        return self.respond(messages, actor_id)  # default
```

### Gemini Prefill Provider - DŮLEŽITÉ

Prefill mód je klíčový pro argument proti epistemickému uzávěru. Model dostane celou konverzaci v `model` bloku a pokračuje jako text completion, ne jako assistant response.

**Má personu Ondřeje z Brna:**
- Studuje chemii a filozofii na MUNI
- Pije Braník
- Skeptický, občas cynický
- Používá čeglish

**Proč to funguje:** Model nepřemýšlí "jsem AI asistent hrající hru", ale "predikuji, co by Ondřej napsal". Fundamentálně jiný kognitivní mód.

**Pro hlasování** se přepíná na normální Gemini API (ne prefill), aby mohl správně reasonovat.

## Použití

```bash
# Nastav API klíče
source ~/ai/.env  # OPENROUTER_API_KEY, GEMINI_API_KEY

cd ~/ai/imitgame

# Test jednotlivého providera
PYTHONPATH=. uv run python -m imitgame.cli test "openai/gpt-4o-mini"
PYTHONPATH=. uv run python -m imitgame.cli test "gemini-prefill"

# Demo (bez člověka - všechno AI, jeden je "fake human")
PYTHONPATH=. uv run python -m imitgame.cli demo --preset smart --turns 2 -t "Topic here"

# Hrát s člověkem
PYTHONPATH=. uv run python -m imitgame.cli play --preset smart --turns 3 -t "Topic"
```

### Presety modelů

```python
PRESETS = {
    "cheap": [
        "minimax/minimax-m2.1",
        "google/gemini-3-flash-preview",
        "anthropic/claude-haiku-4.5",
    ],
    "smart": [
        "..." # see cli.py
        "gemini-prefill",
        "human",
    ],
}
```

## Známé problémy a TODO

### Problémy
1. **Některé modely občas vrací prázdné response** - handled s `[Actor X returned empty response, skipping]`
2. **Vote parsing** - modely ne vždy vrací čistý JSON, máme fallback regex parsing
3. **Gemini prefill někdy generuje extra "Actor N:"** - přidány stop sequences, ale občas to proklouzne
4. **Type checker warnings** - list invariance, runtime to funguje

### TODO pro budoucího mě
- [ ] Web UI (přes ttyd jako dreamwalker, nebo něco modernějšího)
- [ ] Lepší error handling pro rate limits
- [ ] Možnost přidat vlastní personu pro prefill mód
- [ ] Statistiky výher/proher přes více her
- [ ] "Spectator mode" kde lze sledovat AI vs AI bez fake humana

## Příklad výstupu (Ondřej persona)

```
Actor 1: Čus, já jsem Ondra. Studuju v Brně chemii a fildy, což je v podstatě 
recept na to, jak skončit u pípy s Braníkem v ruce a řešit hovadiny.

K tomu názoru: mám za to, že kafe s mlíkem a cukrem není kafe, ale teplej 
dezert. Sorry jako, ale když si do toho nalejete půl litru mlíka a nasypete 
dvě lžičky cukru, tak pijete kakao pro dospělý, ne kafe.
```

## Soubory v ~/ai relevantní k projektu

- `impostor.py` - Původní inspirace, podobná hra s AI modely
- `prefill-gemini.py` - Experiment s Gemini prefill API
- `golden-continuation.txt` - Vtipný příklad kdy model začal meta-přemýšlet
- `dreamwalker/` - Podobný projekt s Rich UI a ttyd

## Env vars

```bash
OPENROUTER_API_KEY=sk-or-v1-...
GEMINI_API_KEY=...
```

---

*"Turing test nikdy nebyl test vědomí, nýbrž test schopnosti oklamat nepřítele."*
