import numpy as np
import matplotlib.pyplot as plt

from operator import add

# Reaction rates
#k1 = np.random.uniform(0, 2)
#k2 = np.random.uniform(0, 2)
#k3 = np.random.uniform(0, 2)
k1 = 1.0
k2 = 1.0
k3 = 1.0
k4 = 1.0

# Initial conditions
#A_conc_0 = np.random.uniform(0, 1) 
#B_conc_0 = np.random.uniform(0, 1)
#C_conc_0 = np.random.uniform(0, 1)
#D_conc_0 = np.random.uniform(0, 1)
A_conc_0 = 1.0
B_conc_0 = 1.0
C_conc_0 = 1.0
D_conc_0 = 1.0


# Storing initial concentrations
concentrations_0 = [A_conc_0, B_conc_0, C_conc_0, D_conc_0]

# Time
time = list(range(0, 15))

# Estimate the step size
N = 10.0
#h = (max(time) - min(time))/N
h = 1/N

# Diff. equations for concentrations
def dAdt(conc, k1, k2, k3, k4):
  concentration = -k1 * conc[0] + (-1 * k2) * conc[0] * conc[2]
  return concentration

def dBdt(conc, k1, k2, k3, k4):
  concentration = k2 * conc[0] * conc[2] - k4 * conc[1]
  return concentration

def dCdt(conc, k1, k2, k3, k4):
  concentration = -k2 * conc[0] * conc[2] + k3 * conc[3]
  return concentration

def dDdt(conc, k1, k2, k3, k4):
  concentration = k1 * conc[0] - k3 * conc[3] + k4 * conc[1]
  return concentration

# Adapting the RK4 function
def RK4(h, function, concentration, kfun1, kfun2, kfun3, kfun4):
  k1 = h * function(concentration, kfun1, kfun2, kfun3, kfun4)
  k2 = h * function([x + 0.5 * k1 for x in concentration], kfun1, kfun2, kfun3, kfun4)
  k3 = h * function([x + 0.5 * k2 for x in concentration], kfun1, kfun2, kfun3, kfun4)
  k4 = h * function([x + k3 for x in concentration], kfun1, kfun2, kfun3, kfun4)
  k = 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
  return k

conc = []
# First reaction, predicted initial concentrations
concA = concentrations_0[0]
concB = concentrations_0[1]
concC = concentrations_0[2]
concD = concentrations_0[3]


# Concentration calculation for each time point
# The resulting change is added to the concentration
for i in range(1, len(time)):
  # Values prev_A, prev_B of the previous iteration
  prev_A = concA
  prev_B = concB
  prev_C = concC
  prev_D = concD
  concentration = [prev_A, prev_B, prev_C, prev_D]
  change = RK4(h, dAdt, concentration, k1, k2, k3, k4)
  concA += change
  change = RK4(h, dBdt, concentration, k1, k2, k3, k4)
  concB += change
  change = RK4(h, dCdt, concentration, k1, k2, k3, k4)
  concC += change
  change = RK4(h, dDdt, concentration, k1, k2, k3, k4)
  concD += change
  # Preservation of concentrations obtained
  conc.append([concA, concB, concC, concD])

# Insert initial concentrations forward
conc.insert(0, concentrations_0)

# Decomposition of concentrations for imaging
concA = []
concB = []
concC = []
concD = []

for i in conc:
  concA.append(i[0])
  concB.append(i[1])
  concC.append(i[2])
  concD.append(i[3])

print("k1:                      ", k1, " A + B -> D")
print("k2:                      ", k2, " A + C -> B")
print("k3:                      ", k3, " D -> C")
print("k4:                      ", k4, " B -> D")
print("Time parameter:        ", time)
print("Step parameter:     ", h)

concA = np.array(concA)
concB = np.array(concB)
concC = np.array(concC)
concD = np.array(concD)

concA[concA<0] = 0
concB[concB<0] = 0
concC[concC<0] = 0
concD[concD<0] = 0


#print("Variation of concentration A:", concA)
#print("Variation of concentration B:", concB)
#print("Variation of concentration C:", concC)
#print("Variation of concentration D:", concD)

# Rezultatų vaizdavimas
plt.plot(time, concA, label='Variation in A concentration')
plt.plot(time, concB, label='Variation in B concentration')
plt.plot(time, concC, label='Variation in C concentration')
plt.plot(time, concD, label='Variation in D concentration')
plt.title("Graph of changing concentrations")
plt.xlabel("t time")
plt.ylabel("Meaning of concentration variation")
plt.legend()
plt.savefig('variation_of_concentration.png')