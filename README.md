# AngleBasis
A Generative Model and Decomposition for Functional Connectivity

<img src='https://github.com/aorliche/AngleBasis/blob/8db7f4629757187157b51d4694f3090ba4eafe95/images/construction.png' alt='construction.png' width='800'>

We decompose FC into a compressed basis plus a residual.

<img src='https://github.com/aorliche/AngleBasis/blob/8db7f4629757187157b51d4694f3090ba4eafe95/images/identifiability.png' alt='identifiability.png' width='800'>

We find the residual has greatly improved identifiability performance.

<img src='https://github.com/aorliche/AngleBasis/blob/8db7f4629757187157b51d4694f3090ba4eafe95/images/prediction.png' alt='prediction.png' width='800'>

The ensemble of basis plus residual is superior in prediction compared to the original FC.

# To Run
Run the following command in the terminal:

```
pip install -r requirements.txt --break-system-packages
```

Then you are all set! (If you don't want to use pip --break-system-packages you need either install requirements yourself or use a virtual environment.)

Try the codes in the notebook directory:

```
cd notebooks
jupyter notebook
```

# Examples

See the paper in the references section.

# References
<a href='https://arxiv.org/abs/2305.10541'>ArXiv paper</a>

@misc{orlichenko2023angle,
      title={Angle Basis: a Generative Model and Decomposition for Functional Connectivity}, 
      author={Anton Orlichenko and Gang Qu and Ziyu Zhou and Zhengming Ding and Yu-Ping Wang},
      year={2023},
      eprint={2305.10541},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}

# Contact
If you have any questions, please contact me at <a href='mailto:aorlichenko@tulane.edu'>Anton Orlichenko aorlichenko@tulane.edu</a>
