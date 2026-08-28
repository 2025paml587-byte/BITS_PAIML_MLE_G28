# ML Engineering Mini-Project: Delivery / Ride ETA Prediction

This project aims to build an end-to-end Machine Learning pipeline for predicting Delivery / Ride Estimated Time of Arrival (ETA) based on various factors. It is structured to demonstrate key MLOps principles including data acquisition, feature engineering, model training, deployment, monitoring, and documentation.

## Project Structure:
- `configs/`: Configuration files for the project.
- `data/`: Raw and processed data files.
- `docs/`: Project documentation and reports.
- `models/`: Trained machine learning models.
- `notebooks/`: Jupyter/Colab notebooks for experimentation and analysis.
- `src/`: Source code for the ML pipeline components (e.g., data processing, model training, API).

## Getting Started:
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Download the DVC-managed data using `dvc pull`.
4. Refer to the `notebooks/` directory for detailed experimentation and development.

## DVC data setup for team members

The data files are stored in the shared Google Drive DVC remote configured in
`.dvc/config`; they are not downloaded by `git clone` or `git pull`.

1. Install the dependencies from `requirements.txt`.
2. Ask the remote owner to share the configured Google Drive folder with the
	Google account you will use for DVC.
3. From the repository root, run:

	```bash
	dvc pull
	```

4. Complete the Google OAuth sign-in in the browser when prompted. Use the
	account that has access to the shared folder.

If `dvc pull` reports that the remote cannot be accessed, verify the signed-in
Google account has access to the Drive folder and run `dvc pull` again. Do not
commit `.dvc/config.local` or OAuth credential files.
