<!-- Day 1 april 1, 2026 -->
### What I did:
- Downloaded Fake and Real News dataset from Kaggle
- Explored data — 44,898 total rows
- Added label column (0=Fake, 1=Real)
- Combined title + text into content column

### What I learned:
- pd.concat() — combining dataframes
- value_counts() — counting labels
- Binary classification concept

### Day 1 continued:
- Text preprocessing done
- Learned: re library, stopwords, NaN handling
- Error faced: AttributeError float has no lower
- Fix: pd.isna() check added

### Phase 3 completed:

#### What I learned:
- TF-IDF — text to numbers conversion
- Logistic Regression — binary classification
- Sigmoid function — 0 to 1 output
- train_test_split — 80/20 rule
- accuracy, precision, recall, f1-score difference
- pickle vs joblib — why joblib better for ML
- Why vectorizer save karna zaroori hai

#### Results:
- Model Accuracy: 99%
- Precision: 99%
- Recall: 99%
- F1 Score: 99%

#### Errors faced:
- Only one class in y_train → ttitle column issue in True.csv
- NaN in content column → fixed with fillna

main things i faced 

Phase 1: Dataset & Environment Setup
Goal: Gather data and prepare the workspace.

Challenge 1: Finding the "Right" Data.

Error: Searching Kaggle for generic "news" didn't give labeled data.

Fix: Specifically targeted the "Fake and Real News Dataset" which provided separate True.csv and Fake.csv files.

Challenge 2: The "Blind" Model Problem.

Logic Error: Realizing that if we just combine the files, the model won't know which is which.

Fix: Manually created a label column: 0 for Fake, 1 for Real.

Challenge 3: The "Ttitle" Typo.

Error: KeyError: 'title' when merging.

Discovery: The True.csv file had a typo in the header called ttitle instead of title.


Fix: Used .fillna() to merge ttitle into the title column before combining. 

Phase 2: Text Preprocessing (Cleaning the Noise)
Goal: Convert human "slang" into clean data for the machine.

Challenge 4: The NLTK Connection Error.

Error: [Errno 11001] getaddrinfo failed while downloading stopwords.

Fix: Manually downloaded the stopwords.zip and placed it in the local nltk_data/corpora/ directory.

Challenge 5: The "Float" Attribute Error.

Error: AttributeError: 'float' object has no attribute 'lower'.

Logic: Some rows in the dataset were empty (NaN). Pandas treats empty cells as floats, which don't have a .lower() method.


Fix: Added a pd.isna(text) check at the start of the clean_text function to return an empty string for null values. 

Phase 3: TF-IDF & Model Training
Goal: Math-heavy part where text becomes numbers.

Challenge 6: The "Single Class" Training Error.

Error: ValueError: This solver needs samples of at least 2 classes.

Discovery: After cleaning empty rows, our y_train only contained 0s. This happened because the True.csv data wasn't being processed correctly.


Fix: Fixed the indexing and ensured the mask was applied to both X and y simultaneously. 

Challenge 7: Memory Efficiency (Pickle vs. Joblib).


Decision: Chose joblib over pickle because the TF-IDF matrix (5,000 features) is a large NumPy-based array, which joblib handles much faster. 

Phase 4: Streamlit Dashboard & Deployment
Goal: Building the "Front-end" for the user.

Challenge 8: The "Tranform" Typo.

Error: AttributeError: 'TfidfVectorizer' object has no attribute 'tranform'.

Fix: Fixed the spelling to transform. (A classic "Developer Eye" error!) 

Challenge 9: The "DRY" (Don't Repeat Yourself) Problem.

Error: Having to copy-paste the clean_text function from train.py into app.py.

Professional Fix: Created utils.py to store the cleaning logic and imported it into both files. This ensures the app cleans text exactly like the training script did. 

Challenge 10: Socket Buffer Space (WinError 10055).

Error: Streamlit crashed because the system lacked buffer space.


Fix: Cleared "zombie" Python processes and restarted the network stack/computer to free up local ports. 

📈 Final Technical Metrics
Model: Logistic Regression (Classification)

Vocabulary: Top 5,000 TF-IDF features.

Accuracy: 99.03%


Precision/Recall: 0.99 (Indicates the model is equally good at catching fakes and verifying real news).

### Phase 5 — Refactoring & Modular Architecture
- **Challenge:** Folder was cluttered with 12+ files in a single directory.
- **Action:** Implemented Modular Architecture (Data/Models/Src split).
- **Learning:** Path management using relative paths (`../`) and the importance of `__init__.py`.
- **Logic:** Separation of Concerns—keeping the "UI" (app.py) separate from the "Logic" (src/).

### Learning Update: The "Why" of .gitignore
- **Concept**: Separation of Data and Logic.
- **Why**: GitHub is for code (recipe), not for raw ingredients (data) or finished products (models).
- **Risk**: Pushing models/data causes repository bloat and security risks.
- **Solution**: Documented the "Training Pipeline" in README so others can reproduce my results.

### Phase 11 — Cloud Native Deployment (Hugging Face + Streamlit)
- **Strategy**: Used Hugging Face for large-scale data hosting (600MB+).
- **Automation**: Created `setup.sh` to automate environment prep and model training in the cloud.
- **Challenge**: Bypassing GitHub's 100MB file limit.
- **Fix**: Used `curl` to pull datasets directly into the Streamlit server instance.
- **Learning**: Mastered the "Cold Start" problem in cloud deployment—ensuring dependencies and models are ready before the first user arrives.

### Phase 12 — UI/UX Engineering
- [cite_start]**Optimization**: Implemented `@st.cache_resource` to prevent redundant model loading. 
- **UX**: Added `st.spinner` and `st.status` for real-time feedback during long-running tasks.
- **Architecture**: Used columns and sidebars to create a professional dashboard layout.

