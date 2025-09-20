\# HMM Viterbi Decoder



A clean, self-contained Viterbi implementation for Hidden Markov Models (HMM).



Given:

\- initial state weights

\- transition weights (state, action → next state)

\- emission weights (state → observation)

\- a sequence of observations/actions



…the script decodes the most likely hidden state sequence and writes it to `data/states.txt`.



---



\## Repo Layout



hmm-viterbi-decoder/

&nbsp; ├─ src/

&nbsp; │   └─ my\_solution.py              # Viterbi + normalization + I/O (paths resolved via pathlib)

&nbsp; ├─ data/

&nbsp; │   ├─ states.txt                  # predicted output (created/overwritten by the script)

&nbsp; │   ├─ state\_weights.txt           # initial state weights

&nbsp; │   ├─ state\_observation\_weights.txt   # emission weights (state, observation)

&nbsp; │   ├─ state\_action\_state\_weights.txt  # transition weights (prev\_state, action, next\_state)

&nbsp; │   └─ observation\_actions.txt     # observation+action sequence (in order)

&nbsp; └─ README.md



> The script resolves file paths relative to the project root, so you can run it from anywhere.



---



\## Run



python src/my\_solution.py



This creates (or overwrites) `data/states.txt` with the predicted sequence, e.g.:



states

3

"A"

"B"

"A"



---



\## Notes



\- Inputs are small and kept under `data/` for clarity.

\- If you don’t want to commit the generated output, add `data/states.txt` to `.gitignore`.



---



\## License



MIT License.



