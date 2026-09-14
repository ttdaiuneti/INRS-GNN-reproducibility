from pathlib import Path
import diagnostic_stream as b
b.SEED=0
b.K_INSERTS=1000
b.CHECKPOINTS=(0,500,1000)
b.OUT=str(Path(__file__).parent/'diagnostic_seed0.json')
b.main()
