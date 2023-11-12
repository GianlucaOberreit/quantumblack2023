import streamlit as st

def set_up(hex_color, max_width=1200, padding_top=1, padding_right=1, padding_left=1, padding_bottom=1, text_color="#FFF", background_color="#0A100D"):
    st.set_page_config(layout="wide")
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container {{
                max-width: {max_width}px;
                padding-top: {padding_top}rem;
                padding-right: {padding_right}rem;
                padding-left: {padding_left}rem;
                padding-bottom: {padding_bottom}rem;
            }}
            .reportview-container .main {{
                color: {text_color};
                background-color: {background_color};
            }}
            .stApp {{
                background-color: {hex_color};
            }}
            .centered {{
                text-align: center;
            }}
            .text {{
                font-family: 'Montserrat', sans-serif;  
                font-weight: bold;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
set_up("#0A100D") # Set background color

def Home():
    # Layout with columns
    col1, col2, col3, col4 = st.columns([1,14,1.2,1])

    with col1:
        # Display small logo
        st.image("./small_logo.png", width=100)

    with col3:
        if st.button('Sign Up'):
            pass  # Perform sign-up action
    with col4:
        if st.button('Log In'):
            pass  # Perform log-in action


    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        empty_space(4)
        st.image("./big_logo.jpg", width=780)
        # Display and center the fixed string using markdown
    empty_space(4)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        fixed_string = "Atmospheric intelligence to \ninform the energy transition"
        st.markdown(f"<div class='centered'><h1 class='text'>{fixed_string}</h1></div>", unsafe_allow_html=True)
        if st.button('Discover it yourself', key=10, use_container_width=True):
            AtmospheR()


    #with col2:
    #    if st.button('AtmospheR', key='1', use_container_width=True):
    #        pass

    #    if st.button('Reveal', key='2',use_container_width=True):
    #        pass  # Action for Button #2

    #    if st.button('TrackeR', key='3',use_container_width=True):
    #        pass  # Action for Button #3

    # Footer
    footer = """
        <div class='centered' style='display: flex; justify-content: center; padding: 10px;'>
            <div style='width: 24px; height: 24px; background: gray; border-radius: 50%; margin: 0 10px;'></div>
            <div style='width: 24px; height: 24px; background: gray; border-radius: 50%; margin: 0 10px;'></div>
            <div style='width: 24px; height: 24px; background: gray; border-radius: 50%; margin: 0 10px;'></div>
            <div style='width: 24px; height: 24px; background: gray; border-radius: 50%; margin: 0 10px;'></div>
        </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

def AtmospheR():




def empty_space(i):
    for _ in range(i):
        st.markdown('#')
if __name__ == '__main__':

    Home()