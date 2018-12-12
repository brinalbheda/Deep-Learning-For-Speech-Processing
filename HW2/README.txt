HW2 folder contains 1 file1 - hw2.py

--Steps to run the code:

1. Change the directory to where the audio files will be saved in the script on line 24.
2. Run hw2.py from terminal.

--What does this code do?

1. Downloading the pure audio signals like TedTalk for test and train inputs and also download white noise for noise input.
2. Mixing clean input with white noise.
3. Applying Hamming window function to input.
4. Converting time domain signal to frequency domain using rfft.
5. Extracting MFCC’s from the frequency domain signals.
6. Converting frequency domain signal to time domain using irfft.
7. Extracting frequency back from MFCC.
8. Constructing the filterbank.
9. Reading the audio files and selecting a part of the input speech.
10.Mixing pure data with noise.
11.Plotting the original signal and processed signal.
12.Splitting the train, test and validation data.
13.Constructing the neural network.
14.Training the neural network.
15.Plotting the convergence graph of epochs versus loss.
16.Reconstructing the audio signal and comparing.
17.Plotting the denoised and reconstructed signal graphs and also the audio files.
