import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold
from scipy.stats.stats import pearsonr
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error

def useruserCF(train_data, test_data, k_val):
    ratings_matrix = csr_matrix((train_data['Rating'], (train_data['UserID']-1, train_data['ProfileID']-1)), 
                            shape=(10000, 10000))
    
    # Rows are UserID's (variables)
    # Columns are ProfileID's (observations)
    ratings_matrix = ratings_matrix.toarray()
    
    correlation_matrix = np.corrcoef(ratings_matrix)
    
    # Collect row means
    row_means = np.zeros(10000)
    for i, row in enumerate(ratings_matrix):
        row_means[i] = np.mean(row[row>0])
        
    k_max = k_val
    output = []
    output_userID = []
    output_profileID = []
    for i in xrange(10000):
        # Grab all the predictions requested for user i
        user_data = test_data[test_data['UserID']==i+1]
        # Get indices of most similar users
        k_neighbors = np.argsort(correlation_matrix[i, :])[::-1]

        for j in user_data['ProfileID']:
            # Rating is adjusted from row mean
            rating = row_means[i]
            k = 0
            k_sum = 0
            normalization = 0
            k_index = 1
            while k < k_max and k_index < 10000:
                if ratings_matrix[k_neighbors[k_index], j-1]:
                    k += 1
                    normalization += abs(correlation_matrix[i, k_neighbors[k_index]])
                    k_sum += correlation_matrix[i, k_neighbors[k_index]]*\
                            (ratings_matrix[k_neighbors[k_index], j-1] - row_means[k_neighbors[k_index]])
                        
                k_index += 1
                
            if normalization==0:
                output.append(rating)
                output_userID.append(i+1)
                output_profileID.append(j)
            else:
                p_aj = rating + k_sum/(normalization*0.95)
                if p_aj < 1:
                    p_aj = 1
                elif p_aj > 10:
                    p_aj = 10
                    
                output.append(p_aj)
                output_userID.append(i+1)
                output_profileID.append(j)
    
    return output, output_userID, output_profileID

def itemitemCF(train_data, test_data, k_val):
    ratings_matrix = csr_matrix((train_data['Rating'], (train_data['UserID']-1, train_data['ProfileID']-1)), 
                            shape=(10000, 10000))
    
    # Rows are UserID's (variables)
    # Columns are ProfileID's (observations)
    ratings_matrix = ratings_matrix.toarray().T
    
    correlation_matrix = np.corrcoef(ratings_matrix)
    
    # Collect column means
    row_means = np.zeros(10000)
    for i, row in enumerate(ratings_matrix):
        row_means[i] = np.mean(row[row>0])
        
    k_max = k_val
    output = []
    output_userID = []
    output_profileID = []
    for i in xrange(10000):
        # Grab all the predictions requested for user i
        user_data = test_data[test_data['UserID']==i+1]
        # Get indices of most similar users

        for j in user_data['ProfileID']:
            k_neighbors = np.argsort(correlation_matrix[j-1, :])[::-1]
            # Rating is adjusted from row mean
            rating = row_means[j-1]
            k = 0
            k_sum = 0
            normalization = 0
            k_index = 1
            while k < k_max and k_index < 10000:
                if ratings_matrix[k_neighbors[k_index], i]:
                    k += 1
                    normalization += abs(correlation_matrix[j-1, k_neighbors[k_index]])
                    k_sum += correlation_matrix[j-1, k_neighbors[k_index]]*\
                            (ratings_matrix[k_neighbors[k_index], i] - row_means[k_neighbors[k_index]])
                        
                k_index += 1
                
            if normalization==0:
                output.append(rating)
                output_userID.append(i+1)
                output_profileID.append(j)
            else:
                p_aj = rating + k_sum/normalization
                if p_aj < 1:
                    p_aj = 1
                elif p_aj > 10:
                    p_aj = 10
                output.append(p_aj)
                output_userID.append(i+1)
                output_profileID.append(j)
                
    return output, output_userID, output_profileID

#Read in ratings and training data
train_data = pd.read_csv('train80.csv', header=None)
test_data = pd.read_csv('test20.csv', header=None)
#idmap_data = pd.read_csv('IDMap.csv')
ratings_data = pd.read_csv('ratings.csv')

train_data.columns = ['UserID', 'ProfileID','Rating']
test_data.columns = ['UserID', 'ProfileID','Rating']


#USER USER CALCULATIONS
pred,uid,pid = useruserCF(train_data, test_data, 5)
df=pd.DataFrame(np.column_stack((np.asarray(uid), np.asarray(pid), np.asarray(pred))))
df.columns = ['UserID', 'ProfileID','Prediction']
result = pd.merge(test_data, df)
np.savetxt("user_user_test20_k5-0.95.csv", result['Prediction'].values, delimiter=",")

pred,uid,pid = useruserCF(train_data, test_data, 20)
df=pd.DataFrame(np.column_stack((np.asarray(uid), np.asarray(pid), np.asarray(pred))))
df.columns = ['UserID', 'ProfileID','Prediction']
result = pd.merge(test_data, df)
np.savetxt("user_user_test20_k20-0.95.csv", result['Prediction'].values, delimiter=",")

pred,uid,pid = useruserCF(train_data, test_data, 50)
df=pd.DataFrame(np.column_stack((np.asarray(uid), np.asarray(pid), np.asarray(pred))))
df.columns = ['UserID', 'ProfileID','Prediction']
result = pd.merge(test_data, df)
np.savetxt("user_user_test20_k50-0.95.csv", result['Prediction'].values, delimiter=",")


#ITEM ITEM CALCULATIONS
pred,uid,pid = itemitemCF(train_data, test_data, 10)
df=pd.DataFrame(np.column_stack((np.asarray(uid), np.asarray(pid), np.asarray(pred))))
df.columns = ['UserID', 'ProfileID','Prediction']
result = pd.merge(test_data, df)
np.savetxt("item_item_test20_k10.csv", result['Prediction'].values, delimiter=",")

pred,uid,pid = itemitemCF(train_data, test_data, 50)
df=pd.DataFrame(np.column_stack((np.asarray(uid), np.asarray(pid), np.asarray(pred))))
df.columns = ['UserID', 'ProfileID','Prediction']
result = pd.merge(test_data, df)
np.savetxt("item_item_test20_k50.csv", result['Prediction'].values, delimiter=",")
