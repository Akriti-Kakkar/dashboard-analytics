import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stats import *
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import locale
import plotly.express as px
from pages._index_analysis import app
import plotly.graph_objects as go
from colorama import Fore, Style
import sys
from termcolor import colored
from IPython.display import display
import html
from streamlit_extras.metric_cards import style_metric_cards
import ast

class comp_analysis:
    def __init__(self, spreadsheet_name: list, sheet: str) -> None:
        self.spreadsheet_name = spreadsheet_name
        self.sheet = sheet
    
    @staticmethod
    def page_config() -> None:
        st.set_page_config(page_title='Dashboard', page_icon='🗺', initial_sidebar_state="expanded")
        st.sidebar.image('htts_fund_logo.png', caption='HTTS Fund')
        st.subheader('📈 Comparative Screener')
        # Inject custom CSS to set the width of the sidebar
        st.markdown(
                """
                <style>
                .main {
                    background: linear-gradient(to bottom right, #f0f4f8, #ffffff);
                }
                </style>
                """,
                unsafe_allow_html=True
            )   
        
    def get_data(self):
        conn = st.connection("gsheets", type=GSheetsConnection)
        capital_data = conn.read(worksheet=self.spreadsheet_name[0])
        cashflow_data = conn.read(worksheet=self.spreadsheet_name[4])
        change_data = conn.read(worksheet=self.spreadsheet_name[1])
        mtm_data_forex = conn.read(worksheet=self.spreadsheet_name[2])
        mtm_data = conn.read(worksheet=self.spreadsheet_name[3])  
        details_data = conn.read(worksheet=self.spreadsheet_name[6])         
        capital_data = capital_data.set_index('Date')
        capital_data.index = pd.to_datetime(capital_data.index)
        cashflow_data = cashflow_data.set_index('Date')
        cashflow_data.index = pd.to_datetime(cashflow_data.index)
        change_data = change_data.set_index('Date')       
        change_data.index = pd.to_datetime(change_data.index)
        baskets_lst1 = capital_data.columns.tolist()
        baskets_lst1 = baskets_lst1[2:]        
        frame1 = change_data[baskets_lst1]
        sum_year = frame1.sum()
        sum_year.name = "PnL"
        sum_year = sum_year.sort_values()
        sort_baskets = sum_year.index.tolist()
        baskets_lst = sort_baskets[::-1]
        active_baskets1 = [col for col in capital_data.columns if 
                          capital_data[col].isna().iloc[-1]!=True]   
        active_baskets1 = active_baskets1[2:]       
        frame11 = change_data[active_baskets1]
        sum_year1 = frame11.sum()
        sum_year1.name = "PnL"
        sum_year1 = sum_year1.sort_values()
        sort_baskets1 = sum_year1.index.tolist()
        active_baskets = sort_baskets1[::-1]
        mtm_data_forex = mtm_data_forex.set_index('Date')
        mtm_data_forex.index = pd.to_datetime(mtm_data_forex.index)
        mtm_data_forex = mtm_data_forex[baskets_lst]
        mtm_data = mtm_data.set_index('Date')
        mtm_data.index = pd.to_datetime(mtm_data.index)
        mtm_data = mtm_data[baskets_lst]
        
        self.details_data_comp = details_data
        self.mtm_data_comp = mtm_data
        self.mtm_data_forex_comp = mtm_data_forex
        self.change_data_comp = change_data
        self.cashflow_data_comp = cashflow_data
        self.capital_data_comp = capital_data
        self.baskets_lst_comp = baskets_lst
        self.active_baskets_comp = active_baskets  
        self.conn_comp = conn   
        
    def connect_index(self):
        obj = app(self.sheet)
        obj.get_data()
        obj.calculate_returns()
        index_data = obj.data
        self.index_data_comp = index_data       
        
    def screener(self):
        tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19, tab20 = st.tabs(
            ["PnL", "Correlation", "MWR (PnL %)", "TWR", "Sharpe Ratio", "Sortino Ratio",
                 "Drawdown", "Loss Days", "Win Days", "Win-Loss Ratio"]
            ) 
        with tab11:
            col_sp1, col_sp2 = st.columns(2)               
            co_lst = pd.MultiIndex.from_tuples(
                [('Inception PnL', 'Basket'), ('Inception PnL', 'S&P'), ('Inception PnL', 'Basket - S&P'),
                ('YTD PnL', 'Basket'), ('YTD PnL', 'S&P'), ('YTD PnL', 'Basket - S&P'),
                ('QTD PnL', 'Basket'), ('QTD PnL', 'S&P'), ('QTD PnL', 'Basket - S&P'),
                ('Last 6 Months PnL', 'Basket'), ('Last 6 Months PnL', 'S&P'), 
                ('Last 6 Months PnL', 'Basket - S&P'), ('Last 60 Days PnL', 'Basket'),
                ('Last 60 Days PnL', 'S&P'), ('Last 60 Days PnL', 'Basket - S&p'),
                ('Last 21 Days PnL', 'Basket'), ('Last 21 Days PnL', 'S&P'), 
                ('Last 21 Days PnL', 'Basket - S&P'), ('MTD PnL', 'Basket'), ('MTD PnL', 'S&P'),
                ('MTD PnL', 'Basket - S&P'), ('Last 1 Day PnL', 'Basket'), 
                ('Last 1 Day PnL', 'S&P'), ('Last 1 Day PnL', 'Basket - S&P')],
                names=['Level 1', 'Level 2'])
            pnl_comp = pd.DataFrame(columns=co_lst, index=self.active_baskets_comp)
            pnl_comp.index.names = ["Active Baskets"]
            sum_inc = self.change_data_comp[self.active_baskets_comp].select_dtypes(
                include="number")
            sum_inc.loc[:, "Quarter"] = sum_inc.index.to_period("Q")
            sum_inc.loc[:, "Month"] = sum_inc.index.to_period("M")
            quarter_comp = sum_inc["Quarter"].max()
            month_comp = sum_inc["Month"].max()
            # Define the date range for the last 6 months
            end_date = sum_inc.index.max()
            start_date = end_date - timedelta(days=182)  # Approximate 6 months
            start_date1 = end_date - timedelta(days=60)
            start_date2 = end_date - timedelta(days=21)            
            pnl_comp[("Inception PnL", "Basket")] = sum_inc.sum().values
            pnl_sum_year = self.change_data_comp[self.active_baskets_comp].select_dtypes(
                include="number"
            )
            pnl_sum_year.loc[:, "Year"] = pnl_sum_year.index.year
            year_comp = pnl_sum_year["Year"].max()
            pnl_sum_year = pnl_sum_year[pnl_sum_year["Year"]==year_comp]
            pnl_sum_quarter = sum_inc[sum_inc['Quarter']==quarter_comp]
            pnl_sum_month = sum_inc[sum_inc['Month']==month_comp]
            pnl_sum_day = sum_inc[sum_inc.index==end_date]
            pnl_comp[("YTD PnL", "Basket")] = pnl_sum_year[self.active_baskets_comp].sum().values
            pnl_comp[("QTD PnL", "Basket")] = pnl_sum_quarter[self.active_baskets_comp].sum().values
            pnl_comp[("MTD PnL", "Basket")] = pnl_sum_month[self.active_baskets_comp].sum().values
            pnl_comp[("Last 1 Day PnL", "Basket")] = pnl_sum_day[self.active_baskets_comp].sum().values
            new_frame_comp = pd.DataFrame(columns=self.active_baskets_comp)
            for x in self.active_baskets_comp:
                new_frame_comp[f"{x}"] = self.capital_data_comp[x] * self.index_data_comp['returns']
            new_frame_comp.loc[:, "Year"] = new_frame_comp.index.year
            new_frame_comp.loc[:, "Quarter"] = new_frame_comp.index.to_period("Q")
            new_frame_comp.loc[:, "Month"] = new_frame_comp.index.to_period("M")
            new_frame_year = new_frame_comp[new_frame_comp["Year"]==year_comp]
            new_frame_quarter = new_frame_comp[new_frame_comp['Quarter']==quarter_comp]
            new_frame_month = new_frame_comp[new_frame_comp["Month"]==month_comp]
            new_frame_day = new_frame_comp[new_frame_comp.index==end_date]
            pnl_comp[("Inception PnL", "S&P")] = new_frame_comp[self.active_baskets_comp].sum().values
            pnl_comp[("YTD PnL", "S&P")] = new_frame_year[self.active_baskets_comp].sum().values
            pnl_comp[("QTD PnL", "S&P")] = new_frame_quarter[self.active_baskets_comp].sum().values
            pnl_comp[("MTD PnL", "S&P")] = new_frame_month[self.active_baskets_comp].sum().values
            pnl_comp[("Last 1 Day PnL", "S&P")] = new_frame_day[self.active_baskets_comp].sum().values
            pnl_comp[("Inception PnL", "Basket - S&P")] = pnl_comp[
                ("Inception PnL", "Basket")] - pnl_comp[("Inception PnL", "S&P")]
            pnl_comp[("YTD PnL", "Basket - S&P")] = pnl_comp[
                ("YTD PnL", "Basket")] - pnl_comp[("YTD PnL", "S&P")]
            pnl_comp[("QTD PnL", "Basket - S&P")] = pnl_comp[
                ("QTD PnL", "Basket")] - pnl_comp[("QTD PnL", "S&P")]  
            pnl_comp[("MTD PnL", "Basket - S&P")] = pnl_comp[
                ("MTD PnL", "Basket")] - pnl_comp[
                    ("MTD PnL", "S&P")
                ]
            pnl_comp[("Last 1 Day PnL", "Basket - S&P")] = pnl_comp[
                ("Last 1 Day PnL", "Basket")] - pnl_comp[("Last 1 Day PnL", "S&P")
            ]
            pnl_comp[("", "Active Baskets")]= pnl_comp.index
            pnl_comp = pnl_comp.reset_index(drop=True)
            pnl_comp[("", "S#")] = pnl_comp.index + 1
            pnl_comp = pnl_comp.set_index(("", "S#"), drop=True)            


            # Filter the DataFrame
            pnl_last_6_months = sum_inc.loc[(sum_inc.index >= start_date) & (sum_inc.index <= end_date)]
            new_frame_6_months = new_frame_comp.loc[(new_frame_comp.index >= start_date) & (new_frame_comp.index <= end_date)]
            pnl_last_60_days = sum_inc.iloc[-60:]
            new_frame_60_days = new_frame_comp.iloc[-60:]
            pnl_last_21_days = sum_inc.iloc[-21:]
            new_frame_21_days = new_frame_comp.iloc[-21:]
            # Calculate column-wise sum
            column_sums = pd.DataFrame(pnl_last_6_months.sum(),
                                       columns=["Last 6 Months PnL"])
            column_sums_60 = pd.DataFrame(pnl_last_60_days.sum(),
                                          columns=["Last 60 Days PnL"])
            column_sums_21 = pd.DataFrame(pnl_last_21_days.sum(),
                                          columns=["Last 21 Days PnL"])
            column_sums_ind = pd.DataFrame(
                new_frame_6_months[self.active_baskets_comp].sum(),
                                           columns=["Last 6 Months PnL"])
            column_sums_ind_60 = pd.DataFrame(
                new_frame_60_days[self.active_baskets_comp].sum(),
                                              columns=["Last 60 Days PnL"])
            column_sums_ind_21 = pd.DataFrame(
                new_frame_21_days[self.active_baskets_comp].sum(),
                                              columns=['Last 21 Days PnL'])
            pnl_comp[('Last 6 Months PnL', 'Basket')] = column_sums.values #placed in terms of index, sequence of index is same as active baskets
            pnl_comp[('Last 6 Months PnL', 'S&P')] = column_sums_ind.values
            pnl_comp[('Last 6 Months PnL','Basket - S&P')] = pnl_comp[
                ('Last 6 Months PnL', 'Basket')] - pnl_comp[
                    ('Last 6 Months PnL', 'S&P')
                ]
            pnl_comp[("Last 60 Days PnL", "Basket")] = column_sums_60.values
            pnl_comp[("Last 60 Days PnL", "S&P")] = column_sums_ind_60.values
            pnl_comp[("Last 60 Days PnL", "Basket - S&P")] = pnl_comp[
                ("Last 60 Days PnL", "Basket")] - pnl_comp[
                    ("Last 60 Days PnL", "S&P")
                ]
            pnl_comp[("Last 21 Days PnL", "Basket")] = column_sums_21.values
            pnl_comp[("Last 21 Days PnL", "S&P")] = column_sums_ind_21.values
            pnl_comp[("Last 21 Days PnL", "Basket - S&P")] = pnl_comp[
                ('Last 21 Days PnL', 'Basket')] - pnl_comp[
                    ('Last 21 Days PnL', 'S&P')
                ]
            
            st.write(f"Last Updated On {end_date}")

            def refine(dataframe, col_pnl):
                dataframe.columns = dataframe.columns.droplevel(0)
                #dataframe['Active Baskets'] = dataframe.index
                #dataframe1 = dataframe.reset_index(drop=True)
                dataframe.columns = ["Active Baskets", f"{col_pnl}"]
                lst = dataframe.columns.tolist()
                #lst1 = ast.literal_eval(lst)
                dataframe = dataframe.sort_values(by=f"{lst[1]}", ascending=True)
                return dataframe           
                   
            def pie_chart(frame_pie, col_head):
                fig = px.pie(
                    frame_pie,
                    names='Active Baskets',
                    values=f"{col_head} PnL",
                    title=f"{col_head} PnL Distribution by Basket",
                    hole=0.3  # Optional: makes it a donut chart
                )
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
            def div_chart(frame_div, col_div):
                # Set color based on sentiment
                colors = ['green' if score >= 0 else 'red' for score in frame_div[f"{col_div} PnL"]]

                # Create the bar chart using Plotly
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=frame_div[f"{col_div} PnL"],
                    y=frame_div["Active Baskets"],
                    orientation='h',
                    marker_color=colors
                ))

                fig.add_vline(x=0, line=dict(color='black', width=1))

                fig.update_layout(
                    xaxis_title=f"{col_div} PnL",
                    yaxis_title='Active Baskets',
                    title=f"{col_div} of Active Baskets" ,
                    plot_bgcolor='rgba(240,244,248,1)',
                    paper_bgcolor='rgba(240,244,248,1)',
                    font=dict(family='Helvetica', size=14, color='black'),
                    xaxis=dict(
                        title_font=dict(size=16, family='Helvetica', color='black'),
                        tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        title_font=dict(size=16, family='Helvetica', color='black'),
                        tickfont=dict(size=12)
                    ),
                    legend=dict(
                        font=dict(size=14, family='Helvetica', color='black'),
                        bgcolor='rgba(255,255,255,0.5)',
                        bordercolor='black',
                        borderwidth=1
                    )   
                )
                


                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            
            data_comp = pd.DataFrame(pnl_comp[[
                ("", "Active Baskets"), ("Inception PnL", "Basket")]].copy())
            data_comp_year = pd.DataFrame(pnl_comp[[
                ("", "Active Baskets"), ("YTD PnL", "Basket")]].copy())
            data_comp_qtd = pd.DataFrame(pnl_comp[[
                ("", "Active Baskets"), ("QTD PnL", "Basket")]].copy())
            data_comp_6m = pd.DataFrame(pnl_comp[[
                ("", "Active Baskets"), ('Last 6 Months PnL', 'Basket')]])
            data_comp1 = refine(data_comp, "Inception PnL")
            data_comp1_ytd = refine(data_comp_year, "YTD PnL")
            data_comp1_qtd = refine(data_comp_qtd, "QTD PnL")  
            data_comp1_6m = refine(data_comp_6m, "Last 6 Months PnL") 
                         
            with col_sp1:
                #pie_chart(data_comp1, "Inception")
                #pie_chart(data_comp1_qtd, "QTD")
                div_chart(data_comp1, "Inception")
                div_chart(data_comp1_qtd, "QTD")
            st.divider()
            st.title("Inception Till Date PnL: Basket Vs S&P")
            
            locale.setlocale( locale.LC_ALL, 'en_CA.UTF-8' )                     
            # Format currency with -$ for negatives
            def format_currency(val):
                if pd.isna(val):
                    return ""
 
                formatted = locale.currency(abs(val), grouping=True)
                return f"-{formatted}" if val < 0 else formatted

            # Apply color coding
            def color_code(val):
                if pd.isna(val):
                    return ''
                return 'color: red;' if val < 0 else 'color: green;'
            
            def styled_data(columns_select):  
                df_1 = pnl_comp[columns_select].copy()
                df_1.columns = df_1.columns.droplevel(0)   
                df_1 = df_1.sort_values(by=["Basket"], ascending=False)    
                df_1 = df_1.reset_index(drop=True)
                df_1.index = df_1.index + 1
                df_total = pd.DataFrame(df_1[["Basket", "S&P"]].sum())
                df_total = df_total.T
                df_total["Basket - S&P"] = df_total["Basket"] - df_total["S&P"]
                df_total["Active Baskets"] = ""
                df_total = df_total[["Active Baskets", "Basket", "S&P", "Basket - S&P"]]
                df_total.index = ["Total"]  
                df = pd.concat([df_1, df_total])         
                # Style the DataFrame
                styled_df = df.style \
                    .format(format_currency, subset=["Basket", "S&P", "Basket - S&P"]) \
                    .applymap(color_code, subset=["Basket", "S&P", "Basket - S&P"]) \
                    .set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#003366'), ('color', 'white'), ('font-weight', 'bold')]},
                        {'selector': 'thead th:first-child', 'props': [('background-color', '#003366'), ('color', 'white')]},
                        {'selector': 'td', 'props': [('background-color', '#f9f9f9')]},
                        {'selector': 'tbody th', 'props': [('background-color', '#003366'), ('color', 'white')]}
                    ]) \
                    .set_properties(**{'text-align': 'center'})
                return styled_df, df_1

            pnl_table1 = styled_data([("", "Active Baskets"),
                                      ("Inception PnL", "Basket"), 
                                      ("Inception PnL", "S&P"), 
                                      ("Inception PnL", "Basket - S&P")])
            st.table(pnl_table1[0])
            pnl_table1_ind1 = pnl_table1[1].copy()
            
            def bar_graph(data_table, title):
                selected_categories = ["Basket", "S&P", "Basket - S&P"]                 
                # Create grouped bar chart
                fig = go.Figure()
                for cat in selected_categories:
                    fig.add_trace(go.Bar(
                        x=data_table["Active Baskets"],
                        y=data_table[cat],
                        name=cat,
                        hovertemplate=f"<b>%{{x}}</b><br>{cat}: %{{y:,.2f}}<extra></extra>"
                    ))

                fig.update_layout(
                    title=f"{title} PnL by Baskets",
                    xaxis_title="Baskets",
                    yaxis_title="PnL",
                    barmode='group',
                    xaxis_tickangle=-45,
                    hovermode="x unified"
                )

                # Display chart
                st.plotly_chart(fig, use_container_width=True)     
            bar_graph(pnl_table1_ind1, "Inception")
            st.divider()
            st.title('Year-Till-Date PnL: Baskets Vs S&P')  
            pnl_table2 = styled_data([
                ("", "Active Baskets"), ("YTD PnL", "Basket"),
                ("YTD PnL", "S&P"),("YTD PnL", "Basket - S&P")
            ])    
            st.table(pnl_table2[0])
            bar_graph(pnl_table2[1], "YTD")
            st.divider()
            st.title("Last 6 Months PnL: Baskets Vs S&P")
            pnl_table3 = styled_data([
                ("", "Active Baskets"), ("Last 6 Months PnL", "Basket"),
                ("Last 6 Months PnL", "S&P"), 
                ("Last 6 Months PnL", "Basket - S&P")
            ])
            st.table(pnl_table3[0])
            bar_graph(pnl_table3[1], "Last 6 Months")
            st.divider()
            st.title("QTD PnL: Baskets Vs S&P")
            pnl_table4 = styled_data([
                ("", "Active Baskets"), ("QTD PnL", "Basket"),
                ("QTD PnL", "S&P"), ("QTD PnL", "Basket - S&P")
            ])
            st.table(pnl_table4[0])
            bar_graph(pnl_table4[1], "QTD")
            st.divider()
            st.title("Last 60 Days PnL: Baskets Vs PnL")
            pnl_table5 = styled_data([
                ("", "Active Baskets"), ("Last 60 Days PnL", "Basket"),
                ("Last 60 Days PnL", "S&P"), ("Last 60 Days PnL", "Basket - S&P")
            ])
            st.table(pnl_table5[0])
            bar_graph(pnl_table5[1], "Last 60 Days")
            st.divider()
            st.title("Last 21 Days PnL: Baskets Vs PnL")
            pnl_table6 = styled_data(
                [("", "Active Baskets"), ("Last 21 Days PnL", "Basket"),
                 ("Last 21 Days PnL", "S&P"), ("Last 21 Days PnL", "Basket - S&P")]
            )
            st.table(pnl_table6[0])
            bar_graph(pnl_table6[1], "Last 21 Days")
            st.divider()
            st.title("MTD PnL: Baskets Vs S&P")
            pnl_table7 = styled_data(
                [("", "Active Baskets"), ("MTD PnL", "Basket"),
                 ("MTD PnL", "S&P"), ("MTD PnL", "Basket - S&P")]
            )
            st.table(pnl_table7[0])
            bar_graph(pnl_table7[1], "MTD")
            st.divider()
            st.title("Last 1 Day PnL: Baskets Vs S&p")
            pnl_table8 = styled_data(
                [("", "Active Baskets"), ("Last 1 Day PnL", "Basket"),
                 ("Last 1 Day PnL", "S&P"), ("Last 1 Day PnL", "Basket - S&P")]
            )
            st.table(pnl_table8[0])
            bar_graph(pnl_table8[1], "Last 1 Day")
            with col_sp2:
                #pie_chart(data_comp1_ytd, "YTD")
                #pie_chart(data_comp1_6m, "Last 6 Months")
                div_chart(data_comp1_ytd, "YTD")
                div_chart(data_comp1_6m, "Last 6 Months")
        with tab12:
            st.image("coming-soon-business-sign-free-vector.jpg")
        with tab13:
            st.image("coming-soon-business-sign-free-vector.jpg")
        with tab14:
            st.image("coming-soon-business-sign-free-vector.jpg")
        with tab15:
            st.image("coming-soon-business-sign-free-vector.jpg")
        with tab16:
            st.image("coming-soon-business-sign-free-vector.jpg")
        with tab17:
            st.image("coming-soon-business-sign-free-vector.jpg")
        with tab18:
            st.image("coming-soon-business-sign-free-vector.jpg")
        with tab19:
            st.image("coming-soon-business-sign-free-vector.jpg")
        with tab20:
            st.image("coming-soon-business-sign-free-vector.jpg")
        
    def main(self):
        self.page_config()
        self.get_data()
        self.connect_index()
        self.screener()
        
    
spreadsheet_name1 = st.secrets["database"]["spreadsheet_name"]
spreadsheet_name = ast.literal_eval(spreadsheet_name1)
sheet = spreadsheet_name[5]
obj = comp_analysis(spreadsheet_name,sheet)
obj.main()
        