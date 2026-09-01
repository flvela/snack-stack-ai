"""Orders page"""
import json

import streamlit as st
from frontend.utils.common import CONFIG_ERROR_MESSAGE, ORDERS_FILE_CONFIG, config_missing_message


st.header("Snack Stack Orders", text_alignment="center")
st.divider()


uploaded_file = st.session_state[ORDERS_FILE_CONFIG]
if uploaded_file is not None:
  orders_dict = json.loads(uploaded_file.getvalue())
  if orders_dict:
    st.dataframe(data=orders_dict)
else:
  if CONFIG_ERROR_MESSAGE in st.session_state:
    st.error(st.session_state[CONFIG_ERROR_MESSAGE])
  st.warning(f"Orders file missing, configure application in sidebar\n. {config_missing_message()}")
