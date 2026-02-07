# Project Contributors

Thank you to everyone who has contributed to the success of this project! Your efforts, whether large or small, are greatly appreciated.

---

## Community & Project Maintainers

* **musikalkemist:** Author and project creator.
* **HimanshuKGP007:** Project reviewer.
* **MLsound:**  Project maintainer.
    - **Refactoring & Maintenance:**
        - Updated `README.md` to include the full course structure and navigation links.
        - Added `CONTRIBUTING.md` and `CONTRIBUTORS.md` guidelines.
        - Unified project structure, folder naming, and refined `.gitignore`.
        - Updated dependencies for Python 3.11 compatibility.
        - Unified folders' names for consistency.
    - **Version Management:** Established and organized the legacy branch to preserve the original course environment for students, ensuring compatibility with legacy video content while moving the main repository to modern standards.
    - **Features:**
        - Implemented automated GTZAN dataset downloader and updated extraction tools.
        - Ensured `StrPath` compatibility in audio data loading.

### Contributors
* **njpau:** Fixed an undefined variable error in the training function within the mlp.py script (PR #1).
* **jungmin-lim:** Detected a typo in a code comment on mlp.py (Issue #2).
GisbertR: Identified a corrupted audio file (jazz.00054.wav) in the GTZAN dataset and proposed using error handling to skip invalid files during preprocessing. (Issue #2)
* **fergarciadlc:** Improved Windows compatibility by replacing invalid characters in folder names and implementing OS-agnostic path splitting for genre mapping (PR #3).
* **dhunstack:**
    - Fixed Librosa compatibility issues by updating deprecated waveplot to waveshow in the preprocessing module (PR #11).
    - Fixed errors in Lecture 12 by implementing audio loading error handling and updating librosa MFCC keyword arguments (PR #12).

---

Want to see your name on this list? Check out our [CONTRIBUTING.md](CONTRIBUTING.md) file to learn how you can help!
