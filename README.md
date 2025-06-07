# mini_project11
This GitHub repo was created exclusively by Tran Nhat Uy Class of ITE10-K69-01 HUST'28, Hanoi, as the solution for the People and Parcel Share A Ride (Large Size) also known as Vehicle Routing Problem with Pickup and Delivery (VRPPD) problem, in the Mini Project no. 11 of the Fundamentals of Optimization IT3052 class no. 159559, managed by Dr. Pham Quang Dung

Description
K taxis (located at point 0) are scheduled to serve transport requests including N passenger requests 1, 2, . . . , N and M parcel requests 1, 2, . . ., M. Passenger request i (i = 1, . . ., N)) has pickup point i and drop-off point i + N + M, and parcel request i (i = 1, . . . , M) has pickup point i + N and drop-off point i + 2N + M. d(i,j) is the travel distance from point i to point j (i, j = 0, 1, . . ., 2N + 2M). Each passenger must be served by a direct trip without interruption (no stopping point between the pickup point and the drop-off point of the passenger in each route). Each taxi k has capacity Q[k] for serving parcel requests. The parcel request i (i = 1, 2, . . ., M) has quantity q[i].
Compute the routes for taxis satifying above contraints such that the length of the longest route among K taxis is minimal (in order to balance between lengths of taxis).
A route of a taxi k is represented by a sequence of points visited by that route: r[0], r[1], . . ., r[Lk] in which r[0] = r[Lk] = 0 (the depot)

"""
Input
Line 1: contains N, M, and K (1 <= N,M <= 500, 1 <= K <= 100)
Line 2: contains q[1], q[2],  . . ., q[M] (1 <= q[i] <= 100)
Line 3: contains Q[1], Q[2], . . . , Q[K] (1 <= Q[i] <= 200)
Line i + 3 (i = 0, 1, . . ., 2N + 2M): contains the i
th
 row of the distance matrix
Output
Line 1: contains an integer K
Line 2k  (k = 1, 2, . . ., K): contains a positive integer Lk
Line 2k + 1 (k = 1, 2, . . ., K): contains a sequence of Lk integers  r[0], r[1], . . ., r[Lk]
Example
Input
3 3 2
8 4 5 
16 16 
0 8 7 9 6 5 11 6 11 12 12 12 13 
8 0 4 1 2 8 5 13 19 12 4 8 9 
7 4 0 3 3 8 4 12 15 8 5 6 7 
9 1 3 0 3 9 4 14 19 11 3 7 8 
6 2 3 3 0 6 6 11 17 11 6 9 10 
5 8 8 9 6 0 12 5 16 15 12 15 15 
11 5 4 4 6 12 0 16 18 7 4 3 4 
6 13 12 14 11 5 16 0 15 18 17 18 19 
11 19 15 19 17 16 18 15 0 13 21 17 17 
12 12 8 11 11 15 7 18 13 0 11 5 4 
12 4 5 3 6 12 4 17 21 11 0 7 8 
12 8 6 7 9 15 3 18 17 5 7 0 1 
13 9 7 8 10 15 4 19 17 4 8 1 0 

Output
2
6
0 5 1 7 11 0
10
0 4 6 10 3 9 12 2 8 0
"""
