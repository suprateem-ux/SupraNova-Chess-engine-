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
```
Ignore last step in non ubuntu systems
## UCI Options

| Option Name | Type | Default | Min | Max | Description |
|-------------|------|---------|-----|-----|-------------|
| Hash        | spin | 32      | 1   | 4096 | Size of the transposition table in MB. |
| Threads     | spin | 1       | 1   | 8    | Number of search threads to use. |
| Multipv     | spin | 1       | 1   | 4    | Number of principal variations (lines) to calculate and show. |

**Notes:**
- `Hash` affects memory usage and search speed.
- `Threads` allows parallel search across CPU cores.
- `Multipv` displays multiple top moves during search; it doesn’t affect move selection.

---

## `go` Command Options

The engine supports several UCI `go` parameters for flexible searching:

| Parameter   | Type | Description |
|-------------|------|-------------|
| wtime       | int  | White’s remaining time in milliseconds. |
| btime       | int  | Black’s remaining time in milliseconds. |
| movetime    | int  | Search for a fixed duration (ms). Overrides time control. |
| depth       | int  | Maximum search depth in plies. |
| nodes       | int  | Maximum number of nodes to search. |
| infinite    | flag | Search indefinitely until a `stop` command is received. | #sorry , it doesnt work but can be bypassed by ponder = false
| mate        | int  | Search for a mate in N moves. |

**Example UCI commands:**

```text
# Set engine options
setoption name Hash value 64
setoption name Threads value 4
setoption name Multipv value 2

# Start searching with different controls
go depth 20
go movetime 5000
go wtime 300000 btime 300000
go infinite
go mate 3
