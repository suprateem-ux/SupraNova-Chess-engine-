# SupraNova Chess Engine

![SupraNova Logo](https://chessengines.blogspot.com/2021/09/first-test-chess-engine-supernova24.html)

**SupraNova** is a hobbyist Python-based chess engine designed to calculate approximately 30,000 nodes per second. It is optimized for analysis and integration with UCI-compatible graphical user interfaces (GUIs).

> ⚠️ **Important Note:** `ponder=true` does **not** work reliably with `go infinite` mode, especially when paired with `stop` or `quit`. It is recommended to always use `ponder=false` for stable operation.

---

## ⚙️ Features

- **UCI-Compatible**: Seamlessly integrates with UCI-compatible GUIs like Arena, Cute Chess, and Lichess bots.  
- **Python Implementation**: Written entirely in Python, making it accessible and modifiable.  
- **Performance**: Capable of calculating around 30,000 nodes per second.  
- **Evaluation Functions**: Includes advanced evaluation metrics for piece-square tables, mobility, king safety, and more.  
- **Search Algorithms**: Implements alpha-beta pruning with a transposition table for efficient search.

---

## 📥 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/suprateem-ux/SupraNova-Chess-engine-
cd SupraNova-Chess-engine-
chmod +x Supranova_(executable)
