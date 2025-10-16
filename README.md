# HMM Viterbi Decoder
A clean, self-contained Viterbi implementation for Hidden Markov Models (HMM).



Given:

- initial state weights

- transition weights (state, action → next state)

- emission weights (state → observation)

- a sequence of observations/actions



…the script decodes the most likely hidden state sequence and writes it to `data/states.txt`.



---







> The script resolves file paths relative to the project root, so you can run it from anywhere.



---



## Run



python src/my\_solution.py



This creates (or overwrites) `data/states.txt` with the predicted sequence, e.g.:



states

3

"A"

"B"

"A"



---





