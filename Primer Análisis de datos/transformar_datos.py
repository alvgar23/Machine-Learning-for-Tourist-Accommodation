# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 20:18:58 2021

@author: alvar
"""

"""
______________________________________________________________________________
En este paso vamos a convertir los datos string en datos enteros manejables.
______________________________________________________________________________
Queremos mapear de forma manual las siguientes variables: 
    hotel, arrival_date_month, meal, distribution_channel, 
    deposit_type, customer_type, reservation_status
______________________________________________________________________________
Queremos convertir de manera automática las siguientes variables: 
    country, reserved_room_type, assigned_room_type, market_segment 
______________________________________________________________________________
Variables que NO hay que convertir en este paso: 
    children, agent, company, is_canceled, lead_time, arrival_date_year, 
    arrival_date_week_number, arrival_date_day_of_month, stays_in_weekend_nights,
    stays_in_week_nights, adults, babies, is_repetead_guest, previous_cancellations,
    previous_bookings_not_canceled, booking_changes, days_in_waiting_list, adr,
    required_car_parking_spaces, total_of_special_requests, reservation_status_date
______________________________________________________________________________
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy


#leemos datos filtrados
df_filtrado = pd.read_csv("hotel_bookings_filtrado.csv")


#para simplificar código
le = LabelEncoder()

#Variables que queremos convertir de manera automática
conv = df_filtrado[["country",
                    "reserved_room_type", 
                    "assigned_room_type",
                    "market_segment"]].values

#las convertimos
for i in range(len(conv[0])):
   conv[:,i] = le.fit_transform(conv[:,i])
   
   
#Creamos un nuevo set de datos, copia del original
df_conv = deepcopy(df_filtrado)


#Guardamos las variables transformadas en el nuevo dataset
df_conv["country"] = conv[:,0]
df_conv["reserved_room_type"] = conv[:,1]
df_conv["assigned_room_type"] = conv[:,2]
df_conv["market_segment"] = conv[:,3]
   

#Si queremos ver a qué valor se ha convertido cada variable:
le.fit(df_filtrado["market_segment"])
print(le.classes_) #el indice nos indica el número al que se transformó


#Variables que queremos mapear
map_hotel = {
    'Resort Hotel': 0,
    'City Hotel': 1
    } 
     
map_admonth = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
    }

#SC = Undefined = No meal package
map_meal = {
    'Undefined': 0,
    'SC': 0,
    'BB': 1,
    'HB': 2,
    'FB': 3
    }

# map_mseg = {
#     'Direct': 0,
#     'Corporate': 1,
#     'Online TA': 2,
#     'Offline TA/TO': 3,
#     'Aviation': 0,
#     'Groups': 5,
#     'Complementary': 6
#     }

map_dchan = {
    'Undefined': 0,
    'Direct': 1,
    'TA/TO': 2,
    'Corporate': 3,
    'GDS': 4
    }

map_dtype = {
    'No Deposit': 0,
    'Non Refund': 1,
    'Refundable': 2
    }

map_ctype = {
    'Contract': 0,
    'Group': 1,
    'Transient': 2,
    'Transient-Party': 3
    }

map_status = {
    'Check-Out': 0,
    'Canceled': 1,
    'No-Show': 2
    }


#mapeamos 
df_conv["hotel"] = df_conv["hotel"].map(map_hotel)
df_conv["arrival_date_month"] = df_conv["arrival_date_month"].map(map_admonth)
df_conv["meal"] = df_conv["meal"].map(map_meal)
# df_conv["market_segment"] = df_conv["market_segment"].map(map_mseg)
df_conv["distribution_channel"] = df_conv["distribution_channel"].map(map_dchan)
df_conv["deposit_type"] = df_conv["deposit_type"].map(map_dtype)
df_conv["customer_type"] = df_conv["customer_type"].map(map_ctype)
df_conv["reservation_status"] = df_conv["reservation_status"].map(map_status)



# Creamos un nuevo archivo csv con estos nuevos datos transformados
df_conv.to_csv("hotel_bookings_enteros.csv", index=False)


df_conv = pd.read_csv("hotel_bookings_enteros.csv")
print(df_conv.info())