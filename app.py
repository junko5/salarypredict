import streamlit as st
import pandas as pd
import numpy as np
import pickle

# モデルのロード
model = pickle.load(open('model.pkl', 'rb'))

# 言語選択とテキストの設定
languages = ['日本語', 'English']

texts = {
    '日本語': {
        'title': '機械学習サンプル：日本企業の月収予測',
        'app_intro': """
            アプリの紹介

            このアプリは、機械学習モデルを活用して、ユーザーが提供する入力データに基づいて予測結果を提供することを目的としたWebアプリケーションです。
            ユーザーは、簡単なインターフェイスを通じてデータを入力し、即座に予測結果を得ることができます。
            なお機械学習モデル作成において、下記サイトのデータを使用しました。
            https://www.e-stat.go.jp/

            ご意見ご感想あれば、弊社サイトの[お問い合わせ](https://create-more.net/contact/)フォームよりお気軽にご連絡ください。
            """,
        'select_language': '言語を選択',
        'input_label': {
            'age': '年齢',
            'job_type': '職種',
            'job': '職種・役職',
            'company_size': '会社の規模',
            'education': '最終学歴',
        },
        'predict_button': '予測',
        'predict_result': 'あなたの入れた条件によると、残業込みで、下記の月収が平均的だと考えられます',
        'predict_note': 'なお条件に合う年収データがありませんでした。かなり異例の出世をされています。',
        'job_type_options': ['事務職', '技術職'],
        'job_options': {
            '事務職': {
                '事務職（役職なし）': 8,
                '主任・シニアスタッフ': 7,
                '係長・チームリーダー': 6,
                '課長代理・セクションリーダー': 5,
                '課長・マネージャー': 4,
                '副部長・次長': 3,
                '部長': 2,
                '支店長・地域統括マネージャー': 1,
            },
            '技術職': {
                'エンジニア・システムエンジニア職（役職なし）': 16,
                '技術主任・シニアエンジニア': 15,
                '技術係長・チームリードエンジニア': 14,
                '技術課長代理・アソシエイトエンジニアリングマネージャー': 13,
                '技術課長・エンジニアリングマネージャー': 12,
                '技術副部長・テクノロジー副ディレクター': 11,
                '技術部長・テクノロジーディレクター': 10,
                '工場長・ファシリティディレクター': 9,
            },
        },
        'size_options': ['50-100人', '100-500人', '500人以上'],
        'education_options': ['大学卒', '高校卒'],
    },
    'English': {
        'title': 'Machine Learning Sample: Predicting Salaries in Japanese Companies',
        'select_language': 'Select Language',
        'app_intro': """
            Application Introduction

            This app is a web application aimed at providing prediction results based on the input data provided by users using machine learning models.
            For feedback, please contact us via our company website's [contact form](https://create-more.net/en/contact/).
            """,
        'input_label': {
            'age': 'Age',
            'job_type': 'Occupation',
            'job': 'Job/Position',
            'company_size': 'Company Size',
            'education': 'Final Education',
        },
        'predict_button': 'Predict',
        'predict_result': 'Based on the conditions you have provided, the following monthly income, including overtime, is considered average:',
        'predict_note': 'There were no annual income data matching your conditions. You have made an exceptionally unusual career advancement.',
        'job_type_options': ['Administrative', 'Technical'],
        'job_options': {
            'Administrative': {
                'Administrative Staff (No position)': 8,
                'Senior Staff': 7,
                'Section Leader': 6,
                'Assistant Division Manager': 5,
                'Division Manager': 4,
                'Deputy Department Head': 3,
                'Department Head': 2,
                'Branch Manager/Area Manager': 1,
            },
            'Technical': {
                'Engineer/System Engineer (No position)': 16,
                'Senior Engineer': 15,
                'Lead Engineer': 14,
                'Associate Engineering Manager': 13,
                'Engineering Manager': 12,
                'Deputy Technical Director/Deputy Technology Director': 11,
                'Technical Director/Technology Director': 10,
                'Plant Manager/Facility Director': 9,
            }
        },
        'size_options': ['50-100 employees', '100-500 employees', '500+ employees'],
        'education_options': ['University Graduate','High School Graduate'],
    }
}

selected_language = st.sidebar.selectbox(texts['日本語']['select_language'], languages)

# 選択された言語に基づくテキストと選択肢
current_text = texts[selected_language]

job_type_options = current_text['job_type_options']
job_options = current_text['job_options']
size_options = current_text['size_options']
education_options = current_text['education_options']

st.title(current_text['title'])
st.markdown("""
    <div style="text-align: center;">
        <img src="https://create-more.net/wp-content/uploads/2021/03/image-e1709000892155.png" alt="年収予測" style="width:50%">
        
    </div>
    <div style="text-align: center;margin:20px;">
        本Webアプリは、こちらへお引越ししました。This web application has been moved here.：<br>
        <a href="https://salary-predict.cms-sample-site.xyz/">https://salary-predict.cms-sample-site.xyz/</a>
    </div>
""", unsafe_allow_html=True)

# 入力
age = st.number_input(current_text['input_label']['age'], min_value=0, max_value=100, value=25, format='%d')

selected_job_type = st.selectbox(current_text['input_label']['job_type'], job_type_options)
selected_job_options = job_options[selected_job_type]
selected_job = st.selectbox(current_text['input_label']['job'], options=selected_job_options)

company_size = st.selectbox(current_text['input_label']['company_size'], options=size_options)

education = st.selectbox(current_text['input_label']['education'], options=education_options)

# エンコーディング処理
job_encoded = selected_job_options[selected_job]
company_size_encoded = size_options.index(company_size) + 1
education_encoded = education_options.index(education) + 1

# 予測ボタン
if st.button(current_text['predict_button']):
    # 特徴量のDataFrameを作成
    input_df = pd.DataFrame([[age, job_encoded, company_size_encoded, education_encoded]],
                            columns=['age_processed', 'job', 'size', 'education'])
    
    # 予測の実行
    prediction = model.predict(input_df)

    # 予測結果の表示
    formatted_salary = "￥{:,.0f}".format(prediction[0])
    st.markdown(texts[selected_language]['predict_result'])
    st.markdown(f"<h2 style='text-align: center; color: black;'>{formatted_salary}</h2>", unsafe_allow_html=True)