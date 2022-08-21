# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 20:17:03 2022

@author: MUSTAFA
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import linalg

# Part a)

# Declare the constants:

L= 1 # Length of the bar in m.
K= 237 # Thermal conductivity in W/(mK).
C= 900 # Specific heat in J/(kgK).
p= 2700 # Density in kg/m^3.

# Declare the time span:
    
t_start= 0
t_end= 100
t_points= 501

# Time step is found by:
    
d_t= (t_end-t_start)/(t_points-1) # -1 for the end-points.

# Declare the position span:
    
x_start= 0
x_end= 1
x_points= 101

# Position step is found by:
    
d_x= (x_end-x_start)/(x_points-1) # -1 for the end-points.

# Declare a matrix full of zeros for the position & time:
    
pos_time= np.zeros((t_points, x_points))

# Note that the spatial index spans i= 0, ..., x_points-1,
# And the temporal index spans k= 0, ..., t_points-1.

# Add the boundary conditions to the position & time matrix (i mean, the entries are already zero but i write the bc's explicitly):

pos_time[0, 0:]= 100   
pos_time[0:, 0]= 0

# Add the initial condition:
    
pos_time[0:, x_points-1]= 0

# Now, using the finite difference approximation, we can fill the matrix of the position & time:
    
for k in range (0, t_points-1): 
    for i in range (1, x_points-1): 
        pos_time[(k+1),i]=pos_time[k, i]+ (K/(C*p))*((d_t)/((d_x)**2))*(pos_time[k, i+1]-2*pos_time[k, i]+pos_time[k, i-1])      
 
# Declare the time matrix (initially full of zeros):
        
time= np.zeros((t_points,x_points))

# Declare the position matrix (initially full of zeros):
    
position= np.zeros((t_points,x_points))

# Fill these matrices for the appropriate time & position values:
    
for k in range (0,t_points):
    for i in range (0, x_points):
        time[k,i]= k*d_t
        position[k,i]= i*d_x
       
# Part b) 

# Plotting the result for the finite difference method:

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

num_surface = ax.plot_surface(time, position, pos_time, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
clb=fig.colorbar(num_surface, shrink=0.5, aspect=5)
clb.set_label('Temperature (K)')

ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('Position (m)', fontsize=10)
ax.set_zlabel('Temperature (K)', fontsize=10)
ax.set_title('Temperature of the Bar vs Time vs Position (Finite Difference Approximation)')

plt.show()

# Part c)

# Note that the plots in Part c) uses Finite Difference Approximation.

# Plotting for approximately t= 1 s:

plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Position vs Temperature at Different Time Values (Finite Difference Approximation)') 

plt.plot(position[50,0:],pos_time[25,0:], 'r', label='at T= 5 s ') 

# Plotting for approximately t= 50 s: 
    
plt.plot(position[50,0:],pos_time[100,0:], 'k', label='at T= 20 s ') 

# Plotting for approximately t= 90 s:

plt.plot(position[50,0:],pos_time[200,0:], 'b', label='at T= 40 s')

plt.legend()
plt.show()

# Here, note that position[i, 0:] as well would do the job where i= 0, ..., 999.

# Part d)

# Plotting the isotherms for the Finite Difference Approximation:

 
fig, ax = plt.subplots()
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Isotherms (Finite Difference Approximation)')
c1 = plt.contourf(time, position, pos_time, 20, cmap='coolwarm')
c2 = plt.contour(time, position, pos_time, 20, cmap='Greys')
clb=plt.colorbar(c1)
clb.ax.set_title('Temperature (K)',fontsize=8)
plt.clabel(c2, inline=True, fontsize=6)
plt.show()

# Part e)

# For the Crank-Nicolson Method, we have to solve a tridiagonal matrix equation.

# For the tridiagonal matrix, we have to declare a unit matrix (100x100 since we have 100 spatial points):

tri_matrix= np.identity(x_points)

# Define the alpha value:
    
alpha= (K/(C*p))*(d_t/((d_x)**2))

# Below is for the main diagonal:
    
for i in range (0, x_points):
    for j in range (0, x_points):
        if i==j:
            if (i,j)!=(0,0) and (i,j)!=(x_points-1, x_points-1):
                tri_matrix[i,j]=1+alpha
 
# Below is for the superdiagonal:
    
for i in range (0, x_points):
    for j in range (0, x_points):
        if i==(j-1):
            if (i,j)!=(0,1):
                tri_matrix[i,j]=-alpha/2
                
# Below is for the subdiagonal:
    
for i in range (0, x_points):
    for j in range (0, x_points):
        if i==(j+1):
            if (i,j)!=(x_points-1,x_points-2):
                tri_matrix[i,j]=-alpha/2
          
# Now, the tridiagonal matrix is set, but we have to solve the matrix equation in the for loop since the unknowns T_k's (k being the time index) are dependent on the time parameter (k).

# Declare the matrix for the position & time for the Crank-Nicolson Method:
    
pos_time_crank= np.zeros((t_points, x_points))

# Declare the boundary conditions:
    
pos_time_crank[0, 0:]= 100   
pos_time_crank[0:, 0]= 0

# Declare the initial condition:
    
pos_time_crank[0:, x_points-1]= 0

# Now, treat the matrix equation as Tx=A (T being the tridiagonal matrix).

# We solve for the initial x vector at t= d_t seconds (at the first time step):

# Therefore, declare the initial A matrix:
    
A= np.zeros((x_points,1)) 

# Below are the boundary conditions:
    
A[0,0]=0
A[x_points-1,0]=0 

# Now, fill the initial A matrix:
    
for i in range (0, x_points):
    if i!=0 and i!=(x_points-1):
        A[i,0]= (alpha/2)*100+(1-alpha)*100+(alpha/2)*100 # This is for t=d_t s(at the first time step).

# Solve for the initial x vector (which contains temperature values at the next instant):
    
x_initial= linalg.solve(tri_matrix,A)  

# We take the transpose of the initial vector (Recall that the time axis that i initially chose are the rows, therefore we take the transpose) and replace it with the 2nd row of the position & time matrix:
    
pos_time_crank[1,0:]=np.transpose(x_initial)
     
for k in range (2, t_points): # Starts from 2 since the first and second row are set already.
    # A_general stands for the A vectors for all t values except at the k= 0 and 1.
    A_general= np.zeros((x_points,1))
    A_general[0,0]=0
    A_general[x_points-1,0]=0 
    for i in range (1, x_points-1): 
        # A_general matrices must be filled:
        A_general[i,0]= (alpha/2)*x_initial[i-1,0]+(1-alpha)*x_initial[i,0]+(alpha/2)*x_initial[i+1,0]
    # Solve for x, take the transpose and update it with the kth row of the position & time matrix:
    x=linalg.solve(tri_matrix,A_general)    
    x_transpose= np.transpose(x)
    pos_time_crank[k,0:]=x_transpose  
    x_initial=x # Update the temperature values for the next k value.
    
# Plot the result for the Crank-Nicolson Method:
    
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

crank_surface = ax.plot_surface(time, position, pos_time_crank, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)  
clb2=fig.colorbar(crank_surface, shrink=0.5, aspect=5)  
clb2.set_label('Temperature (K)') 

ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('Position (m)', fontsize=10)
ax.set_zlabel('Temperature (K)', fontsize=10)
ax.set_title('Temperature of the Bar vs Time vs Position (Crank-Nicolson Method)')

plt.show()

# Below plots uses Crank-Nicolson Method:
    
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Position vs Temperature at Different Time Values (Crank-Nicolson Method)') 

# Plotting for approximately t= 1 s:

plt.plot(position[50,0:],pos_time_crank[25,0:], 'r', label='at T= 5 s ') 

# Plotting for approximately t= 50 s: 
    
plt.plot(position[50,0:],pos_time_crank[100,0:], 'k', label='at T= 20 s ') 

# Plotting for approximately t= 90 s:

plt.plot(position[50,0:],pos_time_crank[200,0:], 'b', label='at T= 40 s')

plt.legend()
plt.show()


# Plotting also the isotherms for the Crank-Nicolson Method:

fig, ax = plt.subplots()
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Isotherms (Crank-Nicolson Method)')
c1 = plt.contourf(time, position, pos_time_crank, 20, cmap='coolwarm')
c2 = plt.contour(time, position, pos_time_crank, 20, cmap='Greys')
clb=plt.colorbar(c1)
clb.ax.set_title('Temperature (K)',fontsize=8)
plt.clabel(c2, inline=True, fontsize=6)
plt.show()

# Part f)

# Exact Calculation:

# Declare the position & time matrix for the exact calculation:
    
pos_time_exact= np.zeros((t_points, x_points))

# Note that the boundary conditions and the initial condition must hold no matter what.

summation= 0 # For the summation over odd n.
for k in range (0, t_points):
    summation=0
    for i in range (0, x_points):
        summation= 0
        for n in range (1, 2000, 2): # Actually, n goes to infinity but this result is obviously approximated up to n=699.
            summation= (400/(n*np.pi))*np.sin(n*np.pi*position[k,i]*L)*np.exp(-((n**2*np.pi**2/L)*time[k,i]*(K/(C*p)))) # 400 comes from the fact that T_0= 100 K.
            pos_time_exact[k,i]+=summation


# Plotting the exact result:
      
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

exact_surface = ax.plot_surface(time, position, pos_time_exact, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)  
clb4=fig.colorbar(exact_surface, shrink=0.5, aspect=5)  
clb4.set_label('Temperature (K)')

ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('Position (m)', fontsize=10)
ax.set_zlabel('Temperature (K)', fontsize=10)
ax.set_title('Temperature of the Bar vs Time vs Position (Analytical Result)')

plt.show()

# Plotting the isotherms as well for the exact result:
    
fig, ax = plt.subplots()
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Isotherms (Analytical Result)')
c1 = plt.contourf(time, position, pos_time_exact, 20, cmap='coolwarm')
c2 = plt.contour(time, position, pos_time_exact, 20, cmap='Greys')
clb=plt.colorbar(c1)
clb.ax.set_title('Temperature (K)',fontsize=8)
plt.clabel(c2, inline=True, fontsize=6)
plt.show()
# Also plotting the temperature vs position at different time values for the exact result:
    
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Position vs Temperature at Different Time Values (Analytical Result)') 

# Plotting for approximately t= 1 s:

plt.plot(position[50,0:],pos_time_exact[25,0:], 'r', label='at T= 5 s ') 

# Plotting for approximately t= 50 s: 
    
plt.plot(position[50,0:],pos_time_exact[100,0:], 'k', label='at T= 20 s ') 

# Plotting for approximately t= 90 s:

plt.plot(position[50,0:],pos_time_exact[200,0:], 'b', label='at T= 40 s')

plt.legend()
plt.show()

# Part g)

# Below code uses the Crank-Nicolson Method for the (!!!) modified (!!!) heat equation.

# Let us define a positive h value:
    
h= 0.3 # in 1/s.

# Also define the temperature of the environment:
    
T_e= 50 # in K.

# For the modified tridiagonal matrix, we have to declare a unit matrix again (100x100 since we have 100 spatial points):

tri_matrix_mod= np.identity(x_points) # _mod stands for the modified heat equation.

# Below is for the main diagonal:
    
for i in range (0, x_points):
    for j in range (0, x_points):
        if i==j:
            if (i,j)!=(0,0) and (i,j)!=(x_points-1, x_points-1):
                tri_matrix_mod[i,j]=1+alpha+h*(d_t/2) # Notice that the main diagonal is now modified.
 
# Below is for the superdiagonal:
    
for i in range (0, x_points):
    for j in range (0, x_points):
        if i==(j-1):
            if (i,j)!=(0,1):
                tri_matrix_mod[i,j]=-alpha/2
                
# Below is for the subdiagonal:
    
for i in range (0, x_points):
    for j in range (0, x_points):
        if i==(j+1):
            if (i,j)!=(x_points-1,x_points-2):
                tri_matrix_mod[i,j]=-alpha/2

# Declare the matrix for the position & time for the Crank-Nicolson Method (for the modified heat equation):
    
pos_time_crank_mod= np.zeros((t_points, x_points))

# For the below boundary and initial conditions, note that they do not change for the modified heat equation.

# Declare the boundary conditions:
    
pos_time_crank_mod[0, 0:]= 100   
pos_time_crank_mod[0:, 0]= 0

# Declare the initial condition:
    
pos_time_crank_mod[0:, x_points-1]= 0

# Now, treat the matrix equation as Tx=A_mod again (T being the tridiagonal matrix, A_mod being the matrix for the modified heat equation).

# We solve for the initial x vector at t= d_t seconds (at the first time step):

# Therefore, declare the initial A matrix:

A_mod= np.zeros((x_points,1)) 

# Below are the boundary conditions:
    
A_mod[0,0]=0
A_mod[x_points-1,0]=0 

# Now, fill the initial A_mod matrix:
    
for i in range (0, x_points):
    if i!=0 and i!=(x_points-1):
        A_mod[i,0]= (alpha/2)*100+(1-alpha-h*(d_t/2))*100+(alpha/2)*100+d_t*T_e*h # This is for t=d_t s(at the first time step), and notice that it is modified since we use modified heat equation.

# Solve for the initial x vector for the modified heat equation (which contains temperature values at the next instant):
    
x_initial_mod= linalg.solve(tri_matrix_mod,A_mod)  

# We take the transpose of the modified initial vector (Recall that the time axis that i initially chose are the rows, therefore we take the transpose) and replace it with the 2nd row of the position & time matrix:
    
pos_time_crank_mod[1,0:]=np.transpose(x_initial_mod)
     
for k in range (2, t_points): # Starts from 2 since the first and second row are set already.
    # A_general_mod stands for the modified A vectors for all t values except at the k= 0 and 1.
    A_general_mod= np.zeros((x_points,1))
    A_general_mod[0,0]=0
    A_general_mod[x_points-1,0]=0 
    for i in range (1, x_points-1): 
        # A_general_mod matrices must be filled:
        A_general_mod[i,0]= (alpha/2)*x_initial_mod[i-1,0]+(1-alpha-h*(d_t/2))*x_initial_mod[i,0]+(alpha/2)*x_initial_mod[i+1,0]+d_t*T_e*h
    # Solve for x, take the transpose and update it with the kth row of the position & time matrix:
    x_mod=linalg.solve(tri_matrix_mod,A_general_mod)    
    x_transpose_mod= np.transpose(x_mod)
    pos_time_crank_mod[k,0:]=x_transpose_mod
    x_initial_mod=x_mod # Update the temperature values for the next k value.
    

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

mod_surface = ax.plot_surface(time, position, pos_time_crank_mod, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)  
clb6=fig.colorbar(mod_surface, shrink=0.5, aspect=5)  
clb6.set_label('Temperature (K)')

ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('Position (m)', fontsize=10)
ax.set_zlabel('Temperature (K)', fontsize=10)
ax.set_title('Temperature of the Bar vs Time vs Position (with $T_{e}$)')
plt.show()

# Plotting the isotherms of the modified heat equation:
    
fig, ax = plt.subplots()
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Isotherms (Modified Heat Equation)')
c1 = plt.contourf(time, position, pos_time_crank_mod, 20, cmap='coolwarm')
c2 = plt.contour(time, position, pos_time_crank_mod, 20, cmap='Greys')
clb=plt.colorbar(c1)
clb.ax.set_title('Temperature (K)',fontsize=8)
plt.clabel(c2, inline=True, fontsize=6)
plt.show()

# Finally, plotting the temperature vs position at different time values with T_e included:

plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Position vs Temperature at Different Time Values (with $T_{e}$)') 

# Plotting for approximately t= 1 s:

plt.plot(position[50,0:],pos_time_crank_mod[25,0:], 'r', label='at T= 5 s ') 

# Plotting for approximately t= 50 s: 
    
plt.plot(position[50,0:],pos_time_crank_mod[100,0:], 'k--', label='at T= 20 s ') 

# Plotting for approximately t= 90 s:

plt.plot(position[50,0:],pos_time_crank_mod[200,0:], 'b', label='at T= 40 s')

plt.legend()
plt.show()
    
# As it is seen in the figure, thermal equilibrium is satisfied between the environment and bar as the time passing by (also notice the homogenous heat exchange between the environment and bar).

# Name: Ahmet Mustafa Baraz
# ID: 21702127
# Title of the Program: 1D Heat Equation


    