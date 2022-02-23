import math
import random
import collections
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from gurobipy import *

def distance(tup1,tup2):
    temp1 = (tup1[0] - tup2[0])**2
    temp2 = (tup1[1] - tup2[1])**2
    result = math.sqrt(temp1 + temp2)
    return result

def generate(n):
    result = []
    for i in range(n):
        temp_x = random.uniform(0,100) #100 is the scale of the study area
        temp_y = random.uniform(0,100)
        temp = (temp_x,temp_y)
        result.append(temp)
    return result

def ordertime():
    result = random.uniform(0,2)
    return result

def cost(node1,node2,num1,num2):
    cost1 = alpha*distance(node1,node2)
    cost2 = beta*(num1-num2)
    result = cost1 + cost2
    return result

def draw():
    plt.scatter(latitude,longitude,s=5,c=(0,0,0),alpha=0.5)
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    for item in E:
        point1 = item[0]
        point2 = item[1]
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        plt.plot([x1,x2],[y1,y2])

    #plt.show()
    plt.savefig('.png')
    plt.close()

def node_plot():
    zip(*D)
    plt.scatter(*zip(*D),marker='x')
    zip(*P)
    plt.scatter(*zip(*D),s=5,c=(0,0,0),alpha=0.5)
    zip(*V)
    plt.scatter(*zip(*D),marker=6)
    plt.savefig('hello.png')
    plt.close()

def tour_dist(trip):
    total_dist = 0
    for i in range(len(trip)-1):
        temp = distance(trip[i],trip[i+1])
        total_dist += temp
    return total_dist

def travel_time(trip):
    result = tour_dist(trip)/speed
    return speed

def two_opt(trip):
    improved = True
    while improved:
        improved = False
        for i in range(1,len(trip)-2):
            for j in range(i+1,len(trip)):
                if j - i == 1:
                    continue
                new_trip = trip[:]
                new_trip[i:j] = trip[j-1:i-1:-1]
                if tour_dist(new_trip) < tour_dist(trip):
                    trip = new_trip
                    improved = True
    return trip

'''================== Input ==================='''

#input
D = generate(150)  #generate random customer nodes
P = generate(200)  # restaurant (pick up nodes) are real world data
depot = []  #depot is know at the beginning
depot = (random.uniform(0,100),random.uniform(0,100))   #generate a random depot
all_nodes = P + D

V = [(random.uniform(0,100),random.uniform(0,100)) for i in range(20)]

capacity = {}
for v in V:
    capacity[v] = 15
#print(V)
#print(capacity)

speed = 60000 #m/h

#real world data: use read.py

'''
zip(*D)
plt.scatter(*zip(*D),marker='x')
plt.savefig('1111')
zip(*P)
plt.scatter(*zip(*D),s=5,c=(0,0,0),alpha=0.5)
plt.savefig('1111')
zip(*V)
plt.scatter(*zip(*D),marker=6)
plt.savefig('1111')
'''
D_x = []
D_y = []
for node in D:
    D_x.append(node[0])
    D_y.append(node[1])

P_x = []
P_y = []
for node in P:
    P_x.append(node[0])
    P_y.append(node[1])

V_x = []
V_y = []
for node in V:
    V_x.append(node[0])
    V_y.append(node[1])

plt.scatter(P_x,P_y,marker = 'x')
#plt.savefig("restaurant.png")

plt.scatter(D_x,D_y,s=5,c=(0,0,0),alpha=0.5)
#plt.savefig("nodes.png")

plt.scatter(V_x,V_y,marker = 6)
plt.savefig("all_nodes.png")

alpha = 1
beta = 0

'''================== Simulate order and time window ==================='''

#each customer has a corresponding restaurant
order_pair = {}  #customer is the key, restaurant is the value
for customer in D:
    restaurant = random.choice(P)
    order_pair[customer] = restaurant

#time window
TW = {}
for customer in D:
    a = random.uniform(5,6)
    b = random.uniform(7,8)
    temp = [a,b]
    TW[customer] = temp

P_TW = {}
for restaurant in P:
     a = 5
     b = 8
     temp = [5,8]
     P_TW[restaurant] = temp


#two-phase assignment approach
#step 1. initialize clusters


'''================== phase 1. cluster ==================='''


#clusters first routing second (Two-phase Approach)

M = 5 #5 vehicles
 #seed point
pair_list = [(k,order_pair[k]) for k in order_pair] #put customer and corresponding restaurant together
#print(V)

#each node has a cost
cluster_cost = {}
for node in (D + P):
    for v in V:
        cost = distance(node,v)
        cluster_cost[(node,v)] = cost
#print(cluster_cost)

#consider that customer and corresponding restaurant must be clustered together

clusters = {}
loads = [0]*len(V)
V_temp = V.copy()
D_temp = D.copy()
load_dict = {}
for v in V_temp:
    load_dict[v] = 0


for node in D_temp:
    cost_min = 100000000000
    flag = 0
    for v in V_temp:
        temp = order_pair[node]
        cost = distance(v,node) + distance(node,temp) + distance(temp,v)
        if cost < cost_min:
            cost_min = cost
            flag = v
    if not flag in clusters:
        clusters[flag] = []
        clusters[flag].append(node)
        clusters[flag].append(order_pair[node])
    else:
        clusters[flag].append(node)
        clusters[flag].append(order_pair[node])

    load_dict[flag] += 1

print('load dict:',load_dict)

while(True):
    for key in clusters:
        if load_dict[key] > capacity[key]:
            #print('oh no')
            temp_dict = {}
            for node in clusters[key]:
                temp_dict[key] = distance(key,node) + distance(node,order_pair[node]) + distance(order_pair[node],key)
            sorted_temp_dict = sorted(dict.items(),key=lambda kv:kv[-1]) #from high to low

            kickout = [sorted_temp_dict[i][0] for i in range(load_dict[key]-capacity[key])]

            for point in kickout:
                temp_dict = {}




for node in D:
    cost_min = 100000000000
    flag = 0
    for v in V:
        temp = order_pair[node]
        cost = distance(v,node) + distance(node,temp) + distance(temp,v)
        if cost < cost_min:
            cost_min = cost
            flag = v
    if not flag in clusters:
        clusters[flag] = []
        clusters[flag].append(node)
        clusters[flag].append(order_pair[node])
    else:
        clusters[flag].append(node)
        clusters[flag].append(order_pair[node])

print('clusters:',clusters)  #k = vehicle, v = node covered by vehicle
print(len(clusters))
V_to_go = list(clusters.keys())
print(V_to_go)



'''================== phase 2. routing ==================='''

#for each cluster, solve TSP
route_list = []
for i in range(len(clusters)):
    #print(V_to_go[i])
    kk = clusters[V_to_go[i]]
    route = [V_to_go[i]]
    flag = 0
    min_cost = 1000000
    for j in range(len(kk)):
        cost = distance(V_to_go[i],kk[j])
        if cost < min_cost:
            min_cost = cost
            flag = kk[j]
    route.append(flag)
    kk.remove(flag)

    while(True):
        flag = 0
        min_cost = 1000000
        dict = {}
        for node in kk:
            #dict = {}
            dict[node] = distance(route[-1],node)
        sorted_dict = sorted(dict.items(),key=lambda kv:kv[1]) #from low to high
        #print(dict)
        #print(sorted_dict)

        #flag = sorted_dict[0][0]

        for kv in sorted_dict:
            if kv[0] in P or order_pair[kv[0]] in route:
                flag = kv[0]
                break
            else:
                continue


            #cost = distance(route[-1],node)
            #if cost < min_cost and node in route:
                #min_cost = cost
                #flag = node

        kk.remove(flag)
        route.append(flag)

        if len(kk) == 0:
            break
    #route.append(V[i])  #back to the depot
    #print(route)
    route_list.append(route)
print('route_list:',route_list)
print(len(route_list))

#plot two routes on the same figure to illustrate as sample.
for i in range(len(route_list[1])-1):
    x1 = route_list[1][i][0]
    x2 = route_list[1][i+1][0]
    y1 = route_list[1][i][1]
    y2 = route_list[1][i+1][1]

    plt.plot([x1,x2],[y1,y2],color='red')
plt.savefig('graphs/route_example.png')
for i in range(len(route_list[4])-1):
    x1 = route_list[4][i][0]
    x2 = route_list[4][i+1][0]
    y1 = route_list[4][i][1]
    y2 = route_list[4][i+1][1]

    plt.plot([x1,x2],[y1,y2],color='green')
plt.savefig('graphs/route_example.png')
plt.close()


for route in route_list:
    c = random.choice(['red','blue','yellow','black','brown','green'])
    for i in range(len(route)-1):
        x1 = route[i][0]
        x2 = route[i+1][0]
        y1 = route[i][1]
        y2 = route[i+1][1]

        plt.plot([x1,x2],[y1,y2],color=c)
    plt.savefig('routes.png')
    plt.savefig('graphs/'+str()+'.png')
plt.close()



