import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import pi
from numpy import linalg as LA

"""
Setting up the problem. The masses of the two galaxies are equal to 1,
and G has been scaled to 1. The test particles have a mass of 10^-6, as
using data that our M_sun ~ 10^-6 * M_Sg-A*
"""

G=1
M1=1
M2=1
Mtest = 10**(-6)

"""
The radii of the rings of test particles around the galaxy and the number of
test particles on each ring.
"""

Number = [12, 0, 18, 0, 24, 0, 30, 0, 36]
R = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]


"""
Lots of data has to be stored to solve the equations of motion of the masses.
Here the lists storing information are set up.
"""

position1 = [] #position of the first galaxy
position2 = [] #position of the perturbing galaxy
testmasspositions = [] #positions of the test masses
positiontest = []  #array containing positions of test masses, updated for each time step
Rvectors = [] #list of vectors from test masses to galaxy 1
Distances = [] #list of distances from test masses to galaxy 1
Rvectors2 = [] #list of vectors from test masses to galaxy 2
Distances2 = [] #list of distances from test masses to galaxy 2
initialdistances = []
#array of initial distances of test masses to galaxy 1. Used to see whether test mass has been disturbed
testpositions = [] #initial test mass positions
testvelocities = [] #list of test mass velocity vectors
testmomenta = [] #list of test mass momentum vectors
testKEs = [] #list of test mass kinetic energies
testPEs = [] #list of test mass potential energies from galaxies 1 and 2
totaltestKEs = [] #total kinetic energy from all of the test masses
totaltestPEs = [] #total potential energy from all of the test masses
testforces = [] #list of resultant force vectors on test masses
KE1 = [] #list of kinetic energies of galaxy 1 for each time step
KE2 = [] #list of kinetic energies of galaxy 2 for each time step
GalPEs = [] #list of potential energy for both galaxies, updated for each time step
galaxydistances = [] #list of distances between two galaxies, updated for each time step


"""Setup of the galaxies:"""

#Initial positions of of each galaxy
G1pos = np.array([0, 0, 0])
G2pos = np.array([15, -15, 0])
galaxyvector = G2pos - G1pos
galaxydistance = LA.norm(galaxyvector)
galaxydistances.append(galaxydistance)


#Initial velocities of each galaxy
G1vel = np.array([0, 0, 0])
G2vel = np.array([0, 0.32, 0])


#Initial momenta of each galax
G1mom = G1vel * M1
G2mom = G2vel * M2


#Initial Energies of each galaxy
G1KE = 0.5 * M1 * (LA.norm(G1vel))**2
G2KE = 0.5 * M1 * (LA.norm(G2vel))**2
GalPE = -G * M1 * M2 / (galaxydistance)


#Initial forces on each galaxy
F12 = - G * M1 * M2 * galaxyvector/(galaxydistance**3)
F21 = - F12


#Certain initial values are stored in lists, to be updated at each time step
position1.append(G1pos)
position2.append(G2pos)

KE1.append(G1KE)
KE2.append(G2KE)
GalPEs.append(GalPE)

"""Setup of the test masses"""

#ith ring of test masses
for i in range(len(R)):

    #jth test mass in the ith ring
    for j in range(Number[i]):

        #trig argument for the initial circular motion of the test particles
        trig_argument = 2*pi*j/Number[i]

        #position vector
        testpos = np.array([R[i] * np.cos(trig_argument), R[i] * np.sin(trig_argument), 0])
        testpositions.append(testpos)
        np.array(testpositions)

        #displacement from galaxy 1
        Rvector = testpos - G1pos
        Rvectors.append(Rvector)
        np.array(Rvectors)

        #distance from galaxy 1
        Distance = LA.norm(Rvector)
        Distances.append(Distance)
        initialdistances.append(Distance)
        np.array(Distances)

        #displacement from galaxy 2
        Rvector2 = testpos - G2pos
        Rvectors2.append(Rvector2)
        np.array(Rvectors2)

        #distance from galaxy 2
        Distance2 = LA.norm(Rvector2)
        Distances2.append(Distance2)
        np.array(Distances2)

        #velocity of test masses
        testvel = np.sqrt(G*M1/R[i]) * np.array([-np.sin(trig_argument), np.cos(trig_argument), 0])
        testvelocities.append(testvel)
        np.array(testvelocities)

        #momentum for test masses
        testmom = testvel * Mtest
        testmomenta.append(testmom)
        np.array(testvelocities)

        #KE for test masses
        testKE = 0.5 * Mtest * (LA.norm(testvel))**2
        testKEs.append(testKE)
        np.array(testKEs)

        #PE for test masses from both galaxies
        testPE = - G * Mtest * ((M1/Distance) + (M2/Distance2))
        testPEs.append(testPE)
        np.array(testPEs)

        #Forces on test masses from galaxies 1 and 2
        Ftest1 = - G * M1 * Mtest * Rvector/(Distance**3)
        Ftest2 = - G * M2 * Mtest * Rvector2/(Distance2**3)
        testforces.append(Ftest1 + Ftest2)

    positiontest.append(testpositions)

totaltestKE = np.sum(testKEs)
totaltestPE = np.sum(testPEs)

totaltestKEs.append(totaltestKE)
totaltestPEs.append(totaltestPE)


"""Solving the equations of motion for the test masses"""

t=0
dt=0.1
DT = int(1/dt)
Tmax = 100
T = 1


"""
Updates the parameters for each time step, time dt after the previous step.
Certain parameters are stored for each time step, to be used in analysis.
This program uses an Euler method to solve the equations of motion.
"""
while t<Tmax:

    testPE = 0



    for i in range(len(Distances)):
        testmomenta[i] = testmomenta[i] + testforces[i] * dt

        testvelocities[i] = testmomenta[i] / Mtest

        testpositions[i] = testpositions[i] + testvelocities[i] * dt
        testmasspositions.append(testpositions[i])

        Rvectors[i] = testpositions[i] - G1pos
        Distances[i] = LA.norm(Rvectors[i])

        totaltestKE += 0.5 * Mtest * (LA.norm(testvelocities[i]))**2

        totaltestPE += - G * Mtest * ((M1/Distances[i]) + (M2/Distances2[i]))
        testPEs.append(testPE)

        testforces[i] = - G * M1 * Mtest * Rvectors[i]/(Distances[i]**3) - G * M2 * Mtest * Rvectors2[i]/(Distances2[i]**3)

    positiontest.append(testmasspositions)
    totaltestKEs.append(totaltestKE)
    totaltestPEs.append(totaltestPE)

    totaltestPE = 0
    totaltestKE = 0

    testmasspositions = []

    #update for galaxies
    G1mom = G1mom + F21*dt
    G2mom = G2mom + F12*dt

    G1vel = G1mom / M1
    G2vel = G2mom / M2

    G1pos = G1pos + G1vel * dt
    G2pos = G2pos + G2vel * dt

    galaxyvector=G2pos-G1pos
    galaxydistance = LA.norm(galaxyvector)
    galaxydistances.append(galaxydistance)

    G1KE = 0.5 * M1 * (LA.norm(G1vel))**2
    G2KE = 0.5 * M1 * (LA.norm(G2vel))**2
    GalPE = -G * M1 * M2 / (galaxydistance)

    F12 = - G * M1 * M2 * galaxyvector/(galaxydistance**3)
    F21 = - F12

    position1.append(G1pos)
    position2.append(G2pos)

    KE1.append(G1KE)
    KE2.append(G2KE)
    GalPEs.append(GalPE)

    t=t+dt

    #update for test masses

positiontest = np.array(positiontest)
position1 = np.array(position1)
position2 = np.array(position2)
KE1 = np.array(KE1)
KE2 = np.array(KE2)
GalPEs = np.array(GalPEs)
totaltestKEs = np.array(totaltestKEs)
totaltestPEs = np.array(totaltestPEs)


"""
Function outputs the total energy of the system, checking how well energy is conserved
"""
def TotalEnergy(KE1, KE2, GalPEs, totaltestKEs, totaltestPEs):
    totalenergy = KE1 + KE2 + GalPEs + totaltestKEs + totaltestPEs

'''

    print("The maximum value of energy is: %f " %(max(totalenergy)))
    print("The minimum value of energy is: %f " %(min(totalenergy)))
    print("The standard deviation of energy values is: %f " %(np.std(totalenergy)))

'''

'''
    plt.subplot(2, 1, 1)
    plt.plot(totalenergy)

    plt.subplot(2, 1, 2)
    plt.plot((totalenergy - totalenergy[0])/totalenergy[0])
    plt.ylabel('Residual fraction')

    plt.savefig('EnergyCons.pdf')
'''



"""
Function outputs the number and fraction of test particles that have been
disrupted significantly by the perturbing galaxy
"""
def TailFraction(initialdistances, Distances):
    TailNumber = 0

    for i in range(len(initialdistances)):

        if Distances[i] > 10:
            TailNumber += 1

    print(
    """The number of test masses whose distance from M1 has increased by at least 1 is:
    %i"""
    %(TailNumber)
    )

    frac_in_tail = TailNumber/len(Distances)

    print(""
    """The fraction of test masses in the tail is %f"""
    %(frac_in_tail)
    )


"""
Function outputs the closest distance from galaxy 1 to galaxy 2 for simulation
"""
def ClosestDistance(galaxydistances):
    print("The closest distance is: %f " %(min(galaxydistances)))


"""
Function outputs snapshots of the evolution of the system
"""
def Snapshot(positiontest, position1, position2, Tmax):

    plt.figure()
    plt.suptitle('Snapshots of test masses at different times', fontweight='bold')

    for i in range(1,10):

        plt.subplot(3, 3, i)
        T = Tmax * i
        plt.plot(positiontest[T, :, 0], positiontest[T, :, 1], marker='o', linestyle='', markersize=1)
        plt.plot(position1[T,0], position1[T,1], marker='o')
        plt.plot(position2[T,0], position2[T,1], marker='o')
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.title('T = %i' %(T*dt))
        plt.xlabel('distance / units', fontsize='5')
        plt.ylabel('distance / units', fontsize='5')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tick_params(labelsize=8)


    plt.subplots_adjust(left=0.2, hspace=1.0)
    plt.savefig('Snapshots.pdf')





"""Function that animates the paths of the galaxies and test masses"""
def FuncAnimation(Distances, positiontest, position1, position2, Tmax):

    fig, ax = plt.subplots()

    line = []

    for i in range(len(Distances)):
            line.append((ax.plot(positiontest[:, i, 0], positiontest[:, i, 1],
            'b', marker='o', linestyle='', markersize=1))[0])
    line.append((ax.plot(position1[:,0], position1[:,1], marker='o'))[0])
    line.append((ax.plot(position2[:,0], position2[:,1], marker='o'))[0])


    def update(num, positiontest, position1, position2, line):
        for i in range(len(Distances)):
            line[i].set_data(positiontest[:, i, 0][num], positiontest[:, i, 1][num])
        line[len(Distances)].set_data(position1[:,0][num], position1[:,1][num])
        line[len(Distances)+1].set_data(position2[:,0][num], position2[:,1][num])


    ani = animation.FuncAnimation(fig, update, len(positiontest[:, 0, 0]),
                                    fargs=[positiontest, position1, position2, line],
                                    interval=25, blit=False)

    plt.gca().set_aspect('equal', adjustable='box')

    ani.save('simulationeuler.mp4')

TailFraction(initialdistances, Distances)
ClosestDistance(galaxydistances)
TotalEnergy(KE1, KE2, GalPEs, totaltestKEs, totaltestPEs)
Snapshot(positiontest, position1, position2, Tmax)
#FuncAnimation(Distances, positiontest, position1, position2, Tmax)
