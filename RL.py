from collections import namedtuple
import numpy as np
import operator
state=namedtuple('State','m n')
all_states=[]
for i in range(4):
    for j in range(12):
        all_states.append(state(i,j))
cliff_states=all_states[1:11]
goal_state=state(0,11)
start_state=state(0,0)
end_terminal=cliff_states+[goal_state]
dlf_reward=-1
cliff_reward=-100
goal_reward=100
alpha=0.25
gamma=0.9
moves={'<':state(0,-1),'>':state(0,1),'^':state(-1,0),'v':state(1,0)}

def valid_moves(st):
    m=[]
    if st in end_terminal:
        return m
    else:
        if st.m  > 0:
            m.append('^')
        if st.m < 3:
            m.append('v')
        if st.n < 11:
            m.append('>')
        if st.n >0 :
            m.append('<')
    return m

def build_q():
    q={}
    for st in all_states:
        m=valid_moves(st)
        q[st]={}
        for i in range(len(m)):
            q[st][m[i]]=0.0
    return q


class Cliffword:
    def __init__(self):
        self.record_list=[]
        self.reward_sum=0
        self.position=state(0,0)
        self.log_dict={}
        self.record_list=[state(0,0)]
        self.Q=build_q()
    def get_moves(self):
        if self.is_terminal():
            return []
        possible_moves=[]
        if self.position.m > 0:
            possible_moves.append('^')
        if self.position.m < 3 :
            possible_moves.append('v')
        if self.position.n > 0:
            possible_moves.append('<')
        if self.position.n < 11 :
            possible_moves.append('>')
        return possible_moves
    def get_reward(self):
        if self.position in cliff_states:
            return -100
        if self.position in goal_state:
            return 100
        else:
            return -1
    def reward_vs_pos(self,pos):
        if pos in cliff_states:
            return -100
        if pos in goal_state:
            return 100
        else:
            return -1
    def move(self,moving):
        self.position=state(self.position.m+moving.m,self.position.n+moving.n)
        self.record_list.append(self.position)
    def is_terminal(self):
        if self.position in end_terminal:
            return True
        else:
            return False
    def random_run(self):
        while self.is_terminal() == False :
            self.move(moves[np.random.choice(self.get_moves())])
    def seek_q(self,s):
        if s in end_terminal:
            return 0
        dic=self.Q[s]
        v=dic.values()
        v.sort()
        return v[-1]
    def train(self):
        s=state(0,0)
        while s not in end_terminal:
            m=valid_moves(s)
            a=np.random.choice(m)
            s_=state(s.m+moves[a].m,s.n+moves[a].n)
            r_=self.reward_vs_pos(s_)
            #a_=alpha*[r_+gamma*]
            q_best=self.seek_q(s_)
            self.Q[s][a]=self.Q[s][a]+alpha*(r_+gamma*q_best+self.Q[s][a])
            s=s_
    def best_run(self):
        s=state(0,0)
        path=[]
        while s not in end_terminal:
            mo=valid_moves(s)
            if len(mo)==0:
                return path
            qm=[self.Q[s][i] for i in (mo)]
            qm=np.array(qm)
            mov=np.random.choice(self.Q[s].keys(),p=qm.ravel())
            s=state(s.m+moves[mov].m,s.n+moves[mov].n)
            path.append(mov)
            #print mo[qm.argmax()]
        print path


clif=Cliffword()
for i in range(1000):
    clif.train()
#print clif.Q[state(m=1,n=11)]
clif.best_run()
