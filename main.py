import pickle


from ta import add_all_ta_features

# In[2]:


import pandas as pd 
import datetime
import yfinance as yf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#import matplotlib.pyplot as plt
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode


# In[3]:


filename = 'classifier_model.sav'
clf = pickle.load(open(filename, 'rb'))


clf2 = pickle.load(open('classifier_w_indicator_model_hist_high_crossed.sav', 'rb'))
# clf2 = pickle.load(open('classifier_w_indicator_model_reversed.sav', 'rb'))


# In[4]:

st.title("Stock Prediction")


today = datetime.date.today()
lastfive = today - datetime.timedelta(days=23)

day = today.strftime ("%d/%m/%Y")
five = lastfive.strftime ("%d/%m/%Y")


# In[5]:


import investpy


# In[6]:
si = pd.read_excel('StockIndustry.xlsx', index_col=0)
si.reset_index(inplace=True)
si_name = si.columns.values.tolist()

stock_df = investpy.get_stocks_overview(country="Bangladesh", 
                        as_json=False, 
                        n_results=1000)


def Hist_high (price_list):
    hist_high_list = list(())
    hasCrossed = list(())
    for i,p in enumerate(price_list):
        high = max(price_list[i:])
        if high == p:
            IsHistHigh = 1
        else:
            IsHistHigh = p/high
        hist_high_list.append(high)
        hasCrossed.append(IsHistHigh)
    
    return hasCrossed


  
# In[7]:

#st = list[[]]

option = list(( stock_df["name"]).unique())



st.header("Only Use One of the Dropbox")
st.subheader("Cross out previous selection before reusing")
#st.caption("Some selected value might still show but wont be a problem, you can reselect like normal")

opt = st.multiselect(
     'Which companies would you like?(can chose multiple)',
     (option))

st.write("OR")

SubOpt = st.multiselect(
     'Which Industry would you like?(can chose only one)',
     (si_name))

if SubOpt:
  opt = list(( si[SubOpt[0]]).dropna().unique())
  SubOpt = list(())

# if st.button('Clear Selection'):
#      opt = list(())
#      SubOpt = list(())



#opt = Ind[SubOpt]

st.write('You selected:', opt)

for index, item in enumerate(opt):
    opt[index] = stock_df.loc[stock_df["name"]==item]["symbol"].values[0]

 
st.write(opt)
num_day = st.number_input('Number of days',1)

invert = st.checkbox('Invert ')

submit = st.button("Submit")

if submit:
  merged = pd.DataFrame()
  
  for s in opt:
      df4 = investpy.get_stock_historical_data(stock= s,
                                        country='Bangladesh',
                                        from_date="01/01/2007",
                                        to_date= day)
      
      df8 = df4[["Close","Open","High","Low","Volume"]]
      df4["VolAvgNDays"] = df4["Volume"].rolling(20).mean()
      df4 = df4[::-1]
      
      #reverse order
#       dfi = df8[::-1]
      dfi = df8
  
      dfi["LP"] = dfi["Close"].shift(+1)
      dfi["Change"] = ((dfi["Close"]-dfi["LP"])/dfi["LP"])
      dfi = dfi[::-1]
      
      HistoricalHigh = Hist_high(list(dfi["Close"]))
      
      dfi["Historical High"] = HistoricalHigh
      
      dfi = dfi[::-1]
      dfi = add_all_ta_features(dfi, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
      dfi = dfi[::-1]
#       st.write(dfi.shape)
#       st.write(dfi)
#       print(dfi)
#       dfi = dfi[dfi['VolAvgNDays'].notna()]
      dfi.reset_index(inplace=True)
      dfi = dfi.dropna()
      dfi2 = dfi.drop(["LP","Date"],axis=1)
#       st.write(dfi2.head())
#       st.write(dfi)
      
      pred2 = clf2.predict(dfi2)
      dfi["IndPred"] = pred2
      dfi["Name"] = s
      
      #Reverse order
#       dfi = dfi[::-1]
      

#       df4["LP"] = df4["Close"].shift(-1)
#       df4["Change"] = ((df4["Close"]-df4["LP"])/df4["LP"])
#       df4 = df4[df4['VolAvgNDays'].notna()]
#       pred1 = clf.predict(df4[["Close","Volume","VolAvgNDays","Change"]])
#       df4["pred"] = pred1
#       df4["Indicator_pred"] = pred2
#       df4["Name"] = s
      
      if dfi.shape[0] < num_day:
            st.write("Sorry "+ str(num_day) + " days of data for this company isnt available")
      else:
          dfi = dfi[::-1]
          dfi['pattern'] = dfi.groupby((dfi.IndPred != dfi.IndPred.shift()).cumsum()).cumcount()+1
#           dfstart = dfi[((dfi.IndPred == 1) & (dfi.IndPred.shift(1) != 1)) | (dfi.IndPred.iat[0]== 1)]
#           dfend = dfi[((dfi.IndPred == 1) & (dfi.IndPred.shift(-1) != 1)) | (dfi.IndPred.iat[-1]==1)]
#           sks = dfstart ["Close"].tolist()
#           dfnew1 = pd.DataFrame()
#           dfnew1 ["end"] = dfend["Close"]
#           st.write(len(sks))
#           st.write(dfnew1.shape)
#           if len(sks) != dfnew1.shape[0]:
#             sks = np.append(sks, 1)
#           dfnew1 ["start"] = sks
#           dfnew1 ["Profit %"] = (((dfnew1["end"] - dfnew1["start"])/dfnew1["start"])*100).astype(int)
#           dfi = dfi.join(dfnew1)
          if invert == False :
            dfi = dfi[::-1]
          df5 = dfi.head(num_day)
          df5 = df5[["Date","Name","IndPred","pattern",
#                      "start","end","Profit %",
                     "Open","High","Low","Close","Historical High","Volume","Change"]]
          df5.reset_index(inplace=True)
          merged = pd.concat([merged, df5], axis=0)
          def aggrid_interactive_table(df: pd.DataFrame):
            """Creates an st-aggrid interactive table based on a dataframe.
            Args:
                df (pd.DataFrame]): Source dataframe
            Returns:
                dict: The selected row
            """
            options = GridOptionsBuilder.from_dataframe(
                df, enableRowGroup=True, enableValue=True, enablePivot=True
            )
            jscode = JsCode("""
                        function(params) {
                            if (params.data.IndPred === 1) {
                                return {
                                    'color': 'white',
                                    'backgroundColor': 'green'
                                }
                            }
                            if (params.data.IndPred === -1) {
                                return {
                                    'color': 'white',
                                    'backgroundColor': 'red'
                                }
                            }
                        };
                        """)  
            gridOptions=options.build()
            gridOptions['getRowStyle'] = jscode
            options.configure_side_bar()
            #options.configure_selection("single")
            
            selection = AgGrid(
                df,
                enable_enterprise_modules=True,
                gridOptions=gridOptions,
                
                            
                theme="dark",
                #update_mode=GridUpdateMode.MODEL_CHANGED,
                allow_unsafe_jscode=True,
            )
            return selection
  selection = aggrid_interactive_table(df=merged)
  @st.cache
  def convert_df(df):
     return df.to_csv().encode('utf-8')


  csv = convert_df(merged)

  st.download_button(
     "Press to Download",
     csv,
     "file.csv",
     "text/csv",
     key='download-csv'
  )
#if selection:
#st.write("You selected:")
#st.json(selection["selected_rows"])

