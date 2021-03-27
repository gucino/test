# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:23:12 2020

@author: Tisana
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:12:03 2019

@author: Tisana
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


#similarity function
def similarity(target_user,existing_user):
    target_user=np.array(target_user,dtype=float)
    existing_user=np.array(existing_user,dtype=float)
    if len(existing_user.shape)==1:
        similarity=cosine_similarity([target_user],[existing_user])[0]
    else:
        similarity=(cosine_similarity([target_user],existing_user))[0]
    return similarity



#function to convert to matrix
def matrix(data_set): # include user id
    mattrix2=[]
    
    #find number of user in this dataset
    array=np.array(data_set)
    arary=array[:,0]
    a=np.unique(arary)
    num_user=len(a)
    for each_user_index in range(0,num_user):
        user_id=each_user_index+1
        row_list=[user_id]
        
        extend_movie_muliply_rating_list=[0]*len(new_movie_list)
        #find movie that this user watch
        for each_list in data_set:
            if each_list[0]==user_id:
                extend_movie_muliply_rating_list[each_list[1]-1]=each_list[2]
        each_user_list=row_list+extend_movie_muliply_rating_list
        mattrix2.append(each_user_list)

    return mattrix2

#function to predict the raating
def rating_prediction(each_new_user,each_movie_index,train_set,num_neighbor):
    #each_new_user should not include user id
    #train_set should not include user id
    
    a=each_new_user[:]
    del a[each_movie_index] #compare the past rating only
     
    #calculate similarity among all existing users(train set)
    b=train_set[:]
    b=np.array(b)
    b=np.delete(b,each_movie_index,1)#compare past rating only
     
    
    similarity_array=similarity(a,b)
    similarity_list=similarity_array.tolist()
    #find similar user
    index_of_similar_user=[]
    similarity_of_similar_user=[]
    copy=similarity_list[:] #to be able to remove once already consider
    
    #if train set contain lesser observation than num_neighbor
    #num_neighbor gonna be all  observation in train set
    if num_neighbor>len(train_set):
        num_neighbor=len(train_set)
    for i in range(0,num_neighbor):
        max_similarity=max(copy)
        index_of_user=similarity_list.index(max_similarity)
        index_of_similar_user.append(index_of_user)
        copy.remove(max_similarity)
        similarity_of_similar_user.append(max_similarity)
    #find rating from each similar user to target user
    rating_list=[]
    for each_user_index in index_of_similar_user:
        rating=train_set[each_user_index][each_movie_index]
        rating_list.append(rating)
    #predict rating for this user   
    #weighted avg of rating by similarity : if total!=0
    total=sum(similarity_of_similar_user)
    if total!=0:
        weighted_list=list((np.array(similarity_of_similar_user))/total)
        predicted_rating=np.multiply(rating_list,weighted_list)
        predicted_rating=sum(predicted_rating)
    else: #do simple average
        predicted_rating=np.mean(rating_list)
    return predicted_rating

#function that give movie id and return rating
def get_rating(movie_id):
    index_index= movie_id_list.index(movie_id) 
    rating=rating_list[index_index]
    return rating

###########################################################################
###########################################################################
###########################################################################
data=pd.read_csv("u_data.csv",header=None).values.tolist()
movie_df=pd.read_csv("u.item.csv",header=None).values.tolist()

new_movie_list=[]
for each_list in movie_df:
    string=each_list[0]
    new_movie_list.append(string.split(sep="|"))
    
#function to realte gen and index in movie list
def gen_to_index(genre):
    i=genre+5
    return i


#dictionary for genre
all_genre={}
genre_list=["unknown","Action","Adventure","Animation","Children","Comedy",
            "Crime","Documentary","Drama","Fantasy",
            "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi",
            "Thriller","War","Western"]
for each_genre in genre_list:
    all_genre[each_genre]={}


#genre and index
genre_and_index={}
for i in range(0,19):
    genre_and_index[i]=genre_list[i]

#for each genre
for i in range(0,19):
    
    #find all movie consider in this genre
    #print("this is ",genre_and_index[i]," genre")
    movie_id_of_this_genre_list=[] #id i str format
    for each_movie in new_movie_list:
        if each_movie[gen_to_index(i)]=='1':
            movie_id_of_this_genre_list.append(each_movie[0])
    
    #calculate the rating for each movie in this genre
    rating_list=[0]*len(movie_id_of_this_genre_list)
    num_rating_list=[0]*len(movie_id_of_this_genre_list)
    for each_review in data:
        
        #if this movie is in this genre
        if str(each_review[1]) in movie_id_of_this_genre_list: 
            
            index=movie_id_of_this_genre_list.index(str(each_review[1]))
            #rating_list[index]+=1
            rating_list[index]+=each_review[2]
            num_rating_list[index]+=1
    #append in dictionary
    for each_movie_id in  movie_id_of_this_genre_list:      
            movie_name=new_movie_list[int(each_movie_id)-1][1]
            #print(movie_name)
            total_rating=rating_list[movie_id_of_this_genre_list.index(each_movie_id)]
            num_rating=num_rating_list[movie_id_of_this_genre_list.index(each_movie_id)]
            all_genre[genre_and_index[i]][movie_name]=total_rating/num_rating
#find maximum rating for each gene
task1=[["genre","movie name"]]
for each_genre in all_genre:
    print("___")
    print("for ",each_genre)
    max_rating=0

    max_rating_movie_list=[]
    for each_rating in all_genre[each_genre].values():
        if each_rating>max_rating:
            max_rating=each_rating
    print("max : ",max_rating)
    for each_rating,movie in zip(all_genre[each_genre].values(),all_genre[each_genre].keys()):
        #print("each rating ",each_rating)
        if each_rating==max_rating:
            max_rating_movie_list.append(movie)
           
    print(max_rating_movie_list)
    task1.append([each_genre]+max_rating_movie_list)

###########################################################################
###########################################################################
###########################################################################

matrix=matrix(data)
rating_prediction_dict_task2={}
#input user id
while True:
    title="Eneter user ID from 1 to "+str(len(matrix))+" : "
    target_user_id=int(input(title))
    if target_user_id>0 and target_user_id<len(matrix)+1:
        break
#convert into matrix

for each_movie_index in range(0,len(new_movie_list)):
    if each_movie_index%10==0:
        print('movie ',each_movie_index)
    num_neighbor=21
    movie_id=each_movie_index+1
    
    '''
    #make matrix for new user (have not watch this movie before)
    matrix_of_new_user=(np.array(matrix))
    matrix_of_new_user=(matrix_of_new_user[matrix_of_new_user[:,movie_id]==0,:]).tolist()
    '''
    matrix_of_new_user=matrix[target_user_id-1]
    
    #if user already watch this movie : no recommendation
    if matrix_of_new_user[movie_id]==0:
        
        #make matrix for existing user (already watch this movie)
        matrix_of_existing_user=(np.array(matrix))
        matrix_of_existing_user=matrix_of_existing_user[:,1:] #ignore username
        matrix_of_existing_user=(matrix_of_existing_user[matrix_of_existing_user[:,each_movie_index]!=0,:]).tolist()
           
        #if no one to test or no data to compare : no pediction
        if len(matrix_of_existing_user)!=0 and len(matrix_of_new_user)!=0:
            #predict rating for each user 
            user_id=matrix_of_new_user[0]
            #create dic for each user : if not exist
            if user_id not in rating_prediction_dict_task2:
                rating_prediction_dict_task2[user_id]={}   
            a=matrix_of_new_user[1:] #ignore user_id
            predicted_rating=rating_prediction(a,each_movie_index,matrix_of_existing_user,num_neighbor)
            rating_prediction_dict_task2[user_id][movie_id]=predicted_rating

#make recommendation to user that give rating for that movie higher than 4
user_recommend_dict_task2={} #sorted from highest rated to lowest
for each_user_id in rating_prediction_dict_task2:
    
    user_recommend_dict_task2[each_user_id]=[]
    movie_id_list=[]
    rating_list=[]
    for each_movie_id_to_recommend in rating_prediction_dict_task2[each_user_id]:
        if rating_prediction_dict_task2[each_user_id][each_movie_id_to_recommend]>=4:
            #recommend
            movie_id_list.append(each_movie_id_to_recommend)
            rating_list.append(rating_prediction_dict_task2[each_user_id][each_movie_id_to_recommend])
    #sort movie base on rating (high to low)
    sorted_movie_list=sorted(movie_id_list,key=get_rating,reverse=True)
    user_recommend_dict_task2[each_user_id]=sorted_movie_list

#filter to have only top 5 movie
while True:
    title="Top K recommendation, Enter value of K from 1 to "+str(len(user_recommend_dict_task2[target_user_id]))+" :"
    top=int(input(title))
    if top>0 and top<len(user_recommend_dict_task2[target_user_id]):
        break

top_5_task2=[]
for each_key,each_value in zip(user_recommend_dict_task2,user_recommend_dict_task2.values()):
    row_list=[each_key]
    for each_movie_id in each_value[:top]:
        movie_index=each_movie_id-1
        movie_name=new_movie_list[movie_index][1]
        row_list.append(movie_name)
    top_5_task2.append(row_list)
print("User ",top_5_task2[0][0]," recommendation")
print("------------------------------------------------")
for each in top_5_task2[0][1:]:
    print(each)
print("------------------------------------------------")