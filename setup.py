import setuptools
setuptools.setup(     
     name="FOD QR",
     author = "Juan E. Peralta",
     version="0.0.1",
     python_requires=">=3.10", 
     requires = ["setuptools", "pyscf", "numpy", "rdkit", "scipy"]
     packages=["niflosic"],
)
