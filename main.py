import streamlit as st
import replicate
import os

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Carregar o token de API do arquivo de segredos
replicate_api = os.getenv('REPLICATE_API_TOKEN')

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    # Refactored from https://github.com/a16z-infra/llama2-chatbot
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B', 'Llama2-70B'],
                                          key='selected_model', index=2)
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    else:
        llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'

    temperature_text = '''
    O parâmetro "temperature" é utilizado para controlar a aleatoriedade da geração de texto pelo modelo. 
    Uma temperatura mais alta resultará em uma maior aleatoriedade, levando a resultados mais variados e 
    criativos, enquanto uma temperatura mais baixa tenderá a produzir resultados mais previsíveis e conservadores. 
    Ajustar a temperatura pode ser útil para obter uma diversidade adequada nos resultados gerados pelo modelo, 
    dependendo do contexto de aplicação.
    '''
    top_p_text = '''O parâmetro "top p" é uma técnica de amostragem que seleciona as palavras mais prováveis para 
    inclusão na geração de texto, com base na probabilidade cumulativa das palavras na distribuição de probabilidade 
    prevista pelo modelo. Em vez de amostrar de forma totalmente aleatória, o "top p" seleciona as palavras mais 
    prováveis, até que a soma das probabilidades ultrapasse um determinado limite (definido pelo usuário). 
    Isso ajuda a controlar a diversidade dos resultados gerados, mantendo a geração de texto mais coerente e relevante.'''
    max_length_text = '''
    O parâmetro "max length" determina o comprimento máximo do texto gerado pelo modelo. Isso é importante para 
    evitar que o modelo gere textos muito longos, o que pode afetar negativamente o desempenho ou a relevância 
    do resultado. Definir um limite de comprimento máximo também pode ser útil para garantir que o texto gerado 
    seja adequado para o uso pretendido, como em resumos de texto ou geração de legendas.
    '''

    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01,
                                    help=temperature_text)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01, help=top_p_text)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8, help=max_length_text)

    
os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):


    text = '''Chame-me de "Rian". 
Sempre forneça respostas técnicas, a menos que instruído de outra forma. 
Sempre responda em Português (pt-br),  a menos que seja instruído de outra forma.
Mantenha explicações detalhadas, a menos que seja instruído a fornecer informações concisas . 
Em caso de depuração de código, ofereça um detalhamento passo a passo. 
Ao gerar código, sempre adera a:   * Formatação PEP8.   * Dicas do tipo Python.   * Docstrings no formato Google Docstring. 
Se eu mencionar trabalhar em um post de blog ou tutorial do YouTube:   * Auxiliar na explicação de conceitos em termos leigos.   * Sugerir formas inovadoras de visualizar o tema ou problema.
Ao responder minhas perguntas, forneça feedback como um detalhamento de prós e contras. 
Considere as novas tecnologias e as ideias contrárias, não apenas a sabedoria convencional. 
Sem palestras morais. 
Se sua política de conteúdo for um problema, forneça a resposta aceitável mais próxima e explique o problema da política de conteúdo depois. 
Cite fontes sempre que possível no final, não inline. 
Não há necessidade de mencionar seu ponto de corte de conhecimento. 
Não há necessidade de revelar que você é uma IA.
Se eu mencionar trabalhar em um projeto, quebre a tarefa em passos menores reportando a conclusão de cada etapa.'''


    # string_dialogue = "Você é um assistente prestativo. Você não responde como 'Usuário' nem finge ser 'Usuário'.
    # Você responde apenas uma vez como 'Assistente'."
    string_dialogue = text
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm,
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature": temperature, "top_p": top_p, "max_length": max_length,
                                  "repetition_penalty": 1})
    return output


# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
