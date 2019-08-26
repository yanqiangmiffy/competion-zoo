def missing_data(data):
    """
    data:dataframe，展示每列确实值
    """
    total=data.isnull().sum()
    percent=(data.isnull().sum()/data.isnull().count()*100)
    tt=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
    types=[]
    for col in data.columns:
        dtype=str(data[col].dtype)
        types.append(dtype)
    tt['Types']=types
    return(np.transpose(tt))
