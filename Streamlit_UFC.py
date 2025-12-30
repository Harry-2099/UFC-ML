import joblib
from pathlib import Path
import torch
import torch.nn as nn
from xgboost import XGBClassifier
import pandas as pd 
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import StandardScaler
                                    ########### LOAD MODELS ##########
BASE_DIR = Path(__file__).parent
#print("BASE PATH:::",BASE_DIR)

models_dir = BASE_DIR/ "models"
# --- Logistic Regression ---
log_model = joblib.load(models_dir / "ufc_logreg.pkl")

# --- XGBoost ---
xgb_model = XGBClassifier()
xgb_model = joblib.load(models_dir / "ufc_xgb.pkl")

# --- Neural Net ---
class Logit_NeuralNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 18),
            nn.ReLU(),
            nn.Linear(18, 1),
        )

    def forward(self, x):
        return self.net(x)
# SCALER 
scaler = joblib.load(models_dir / "ufc_scaler.pkl")

#must match training feature count
input_dim = 54  
NN_model = Logit_NeuralNet(input_dim)
state = torch.load(models_dir / "ufc_nn.pt", map_location="cpu")
NN_model.load_state_dict(state)
NN_model.eval()
### Ensemble
meta_model = joblib.load(models_dir / "ufc_EN_stacker.pkl")

                            ############# Streamlit Section ###########

import streamlit as st

# INIT STREAMLIT STATES
if "ran_models" not in st.session_state:
    st.session_state.ran_models = False

if "show_table" not in st.session_state:
    st.session_state.show_table = False
    
if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "page" not in st.session_state:
    st.session_state.page = "Welcome!"

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

#title
#st.title("Machine Learning Approach to UFC")
## FOR CENTERING
left, center, right = st.columns([1, 6, 1])

#navigation
pages = ["Welcome!", 
         "How to", "Get Started: Predict & Visualize Outputs","Modeling Table: Detailed Model Outputs","How it Works"]

with st.sidebar.expander("Navigation", expanded=True):
    page = st.radio(
        "Go to",
        pages,
        index=pages.index(st.session_state.page)
    )
st.session_state.page = page


if page == "Welcome!":
    #st.subheader("Welcome!")
    with center:
        st.title("UFC Predictions with Machine Learning")

        st.markdown(
        "<h3 style='text-align:center;'>"
        "Machine Learning–Driven Probability Modeling"
        "</h3>",
        unsafe_allow_html=True)
        st.caption(
            "<h3 style='text-align:center;'>"
            "No Hype, Just Probabilistic Modeling"
            "</h3>",
            unsafe_allow_html=True
            )
    st.divider()
    st.markdown("""   
    #### - Quantify UFC Outcomes with Machine Learning Models. 
    #### - Check out the **How to** section if you've never been here before or jump right to the Get Started Section 
    #### - Modeling Procedure, Methodology, and Data aquasitition Outlined in the How It Works Section""")
    
elif page == "How to":
    st.subheader("**How to**: A Few Clicks to Predict")
    st.markdown("""
    The methodolgy used in this project allows for predictions to be made on entire UFC cards/events by simply entering a link below.
    Naviagate to the offcial ufc statistics page located here http://ufcstats.com/statistics/events/upcoming and click on the event you'd
    like to get predictions for. Copy the url, paste into box below, and hit Run Models. Done! 🥊
    """)
elif page == "Get Started: Predict & Visualize Outputs":
    st.subheader("Paste, Scrape, & Predict 🤖")
    #st.write("State right now:", st.session_state.ran_models)
    #CHECK URL FUNCTIONs
    #UFC URL
    def is_valid_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in ("http", "https")
                and parsed.netloc == "ufcstats.com"
                and parsed.path.startswith("/event-details/")
            )
        except Exception:
            return False
        
    with st.form("url_form"):
        url = st.text_input("### Paste fight link")
        run = st.form_submit_button("Run Models")

    if run:
        if not is_valid_url(url):
            st.error(
            "That link doesnt seem correct pal 🙃 \n Read the **How To** and enter the correct URL buddy 👎.\n\n" 
            "Make sure you are using an **upcoming UFCStats event link**.")
        else:
            with st.spinner("Fetching page..."):
                response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
            st.success("Models Ran Successfully")

        ### Scrapping the link now
            #USING TRY TO CATCH INCORRECT URLS FURTHER
            try:
                soup = BeautifulSoup(response.text, "html.parser")

                # get all fighter links
                name_elements = soup.select(".b-link_style_black")

                # Extract text and filter out "View Matchup"
                fighter_names = [el.get_text(strip=True) for el in name_elements if "View" not in el.get_text()]

                # Create empty lists for Name and Opponent
                names = []
                opponents = []

                # Iterate in pairs
                for i in range(0, len(fighter_names), 2):
                    try:
                        names.append(fighter_names[i])
                        opponents.append(fighter_names[i+1])
                    except IndexError:
                        # in case there's an odd number of fighters
                        opponents.append(None)

                # Create DataFrame
                event = pd.DataFrame({
                    "Name": names,
                    "Opponent": opponents
                })
            except ValueError as e:
                st.error("This page could not be parsed as a UFC event.\n\n"
                         "Make sure you are using an **upcoming UFCStats event link**.")
                st.caption(f"Details: {e}")
                st.stop()
            
            #double check
            if event.shape[0] == 0:
                st.error("This page could not be parsed as a UFC event.\n\n"
                         "Make sure you are using an **upcoming UFCStats event link**.")
                st.caption(f"Details: {e}")
                st.stop()

            #df ======> historic data 

            df = pd.read_csv(BASE_DIR/"final_ufc_stats.csv")
            df = df.drop(columns="Unnamed: 0" )
            # Merge fighter stats
            event_stats = event.merge(df, on="Name")

            # Merge opponent stats
            event_stats = event_stats.merge(df.add_prefix("opp_"), left_on="Opponent", right_on="opp_Name")
            # 1. Get all numeric columns except Y
            numeric_cols = [
                col for col in event_stats.columns
                if event_stats[col].dtype in ['int64', 'float64']
                and "STANCE" not in col
            ]
            numeric_cols = [ x for x in numeric_cols if x not in ['KD', 'STR', 'TD', 'Sub', 'round'] ]
            print(numeric_cols)
            # 2. Separate fighter stats and opponent stats
            f1_cols = sorted([col for col in numeric_cols if not col.startswith("opp_")])

            # 2. Separate fighter stats and opponent stats
            f2_cols = sorted([col.replace("opp_", "") for col in numeric_cols if col.startswith("opp_")])

            # 3. Build real column names for subtraction
            f2_full_cols = ["opp_" + col for col in f2_cols]

            # 4. Compute deltas
            delta_df = event_stats[f1_cols].values - event_stats[f2_full_cols].values

            # 5. Create delta names
            delta_names = [f"delta_{col}" for col in f1_cols]

            # 6. Assign back into dataframe
            event_stats[delta_names] = delta_df

            ## Make stance int
            stance_cols = [c for c in event_stats.columns if "STANCE" in c]
            event_stats[stance_cols] = event_stats[stance_cols].astype(int)

            # Add a readable fight index before dropping columns
            event_stats["matchup"] = event_stats["Name"] + " vs " + event_stats["Opponent"]
            # Columns to drop from feature matrix
            drop_cols = [
                col for col in event_stats.columns
                if event_stats[col].dtype == "object"          # drop all string columns
                or col in ['KD', 'STR', 'TD', 'Sub', 'round']  # drop raw fight stats
            ]

            drop_cols += ["DOB", "opp_DOB", "matchup"]   # EXCLUDE matchup from X
            # Create model input
            x_new = event_stats.drop(columns=drop_cols)


                                ###########  PREDICTIONS and MODELING #########
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            #move model to gpu
            NN_model = NN_model.to(device)
            # Convert x_new for NN
            # ----------------------------
            x_new_scaled = scaler.transform(x_new)
            x_new_t = torch.tensor(x_new_scaled, dtype=torch.float32).to(device)

            ###### Get Preds
            # Logistic Regression
            log_model_prob = log_model.predict_proba(x_new)[:, 1]

            #  XGBoost
            xgb_prob = xgb_model.predict_proba(x_new)[:, 1]

            # Neural Net
            NN_model.eval()
            with torch.no_grad():
                NN_logit =NN_model(x_new_t)
                NN_prob = torch.sigmoid(NN_logit).cpu().numpy().flatten()

                                ################ ENSEMBLE ############
            X_meta_new = np.column_stack([log_model_prob, xgb_prob, NN_prob])
            # Ensemble (meta-model) predictions

            ### USING META
            new_prob = meta_model.predict_proba(X_meta_new)[:, 1]
            new_pred = (new_prob >= 0.5).astype(int)

            # Save to event_stats
            event_stats["Ensemble Pred"] = new_pred
            event_stats["Ensemble Prob"] = new_prob
            event_stats["Logistic Prob"] = log_model_prob
            event_stats["XGB Prob"] = xgb_prob
            event_stats["Neural Net Prob"] = NN_prob

            #clean data frame
            event_stats["Predicted Winner"] = np.where(
                event_stats["Ensemble Pred"] == 1,
                event_stats["Name"],
                event_stats["Opponent"]
            )
            pred_cols = [col for col in event_stats if "Prob" in col or "Pred" in col and "Predicted" not in col]
            #print(pred_cols)
            pred_cols = ["matchup","Predicted Winner","Name"] + pred_cols
            #print(pred_cols)
            predictions = event_stats[pred_cols]
            predictions = predictions.rename(columns = {
                "Name":"Modeled Fighter"
            })

            st.session_state.predictions = predictions
            #make state true now
            st.session_state.ran_models = True
            st.session_state.show_table = False  # reset on new run

            ##################################### PLots #####################################
            #handeling dark mode/ light mode
            def is_dark_theme() -> bool:
                #get the option from streamlit
                base = st.get_option("theme.base")
                # or checks for truthy -> non null or empty. Check if dark -> return boolean
                return (base or "light").lower() == "dark"
            
            prob_cols = [col for col in event_stats if "Prob" in col in col and "Predicted" not in col]
            #print(pred_cols)
            prob_cols = ["Name","Opponent"] + prob_cols
            probs = event_stats[prob_cols]
            import matplotlib.pyplot as plt

            for _, row in probs.iterrows():
                F1 = str(row["Name"])
                F2 = str(row["Opponent"])
                p1 = float(row["Ensemble Prob"])
                p2 = 1 - p1

                pick = F1 if p1 >= 0.5 else F2

                fig, ax = plt.subplots(figsize=(8, 1.8))

                y = [F1, F2]
                win  = [p1, p2]
                loss = [1 - p1, 1 - p2]
                #BARS
                # set text colors
                dark = is_dark_theme()
                themed_text = "white" if dark else "black"
                themed_bar = "white" if dark else "bisque"
                #colors conditonal on winner
                colors = ["blue" if w >= 0.5 else "firebrick" for w in win]
                #STACKED BARS
                #win part bar
                ax.barh(y, win, height=0.55, color=colors)
                #lose part bar
                ax.barh(y, loss, height=0.2, left=win, color=themed_bar)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Win Probability",color = themed_text)
                ax.set_title(f"{F1} vs {F2} | Predicted Pick: {pick}", fontsize=12,color = themed_text)

                # Transparent background
                #ax.set_facecolor("none")
                # fig.patch.set_alpha(0)
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                ax.tick_params(axis="y",colors=themed_text, length=0)
                ax.tick_params(axis="x", colors=themed_text)

                plt.tight_layout()
                st.pyplot(fig,transparent = True)

# NAVIGATION MODELING TABLE
elif page == "Modeling Table: Detailed Model Outputs":
    st.subheader("Modeling Table")
    st.markdown(
        """
        #### This Table allows you to look at the modeled probabilites for all 4 models yourself.     
        """
    )
    #check if models ran
    if not st.session_state.get("ran_models", False):
        st.info("Run the models first in **Get Started**.")
    else:
        st.dataframe(
            st.session_state.predictions,
            use_container_width=True
        )

    # if st.session_state.get("show_table", False):
    #     st.write("📊 Showing Table")
    #     st.dataframe(st.session_state.predictions, use_container_width=True)

elif page == "How it Works":
    st.header("How it Works")
    st.subheader("Part One: Web Scrapping")
    st.write("Explain methodology........")
    st.subheader("Part Two: Data Manipulation, Cleaning, & Feature Engineering")
    st.write("Explain methodology........")
    st.subheader("Modeling: Logistic Regression, XG Boost & Feedforward Neural Network -> Ensemable Model")
    st.write("Explain methodology........")