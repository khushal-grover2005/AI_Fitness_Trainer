# AI Fitness Trainer 🤖💪

Real-time exercise form analysis using computer vision and pose estimation.

## 🏋️‍♂️ Exercises
- **Lunges** (`lunges.py`) - Side view with balance tracking
- **Squats** (`squat.py`) - Depth analysis and form correction  
- **Push-ups** (`pushup.py`) - Upper body form monitoring

## 🚀 Quick Start

### Installation
Prerequisite: You must have Anaconda installed on your machine
```bash
# 1. Clone the repository
git clone https://github.com/khushal-grover2005/AI_Fitness_Trainer.git
cd AI_Fitness_Trainer

# 2. Create and activate a Conda environment (Python 3.11 recommended)
conda create -n ai_trainer_compat python=3.11
conda activate ai_trainer_compat

# 3. Install the required packages
pip install -r requirements.txt

# 4. Run an exercise script
python lunges.py (side-view)
# or
python pushup.py (front-view)
# or
python squat.py (front-view)

