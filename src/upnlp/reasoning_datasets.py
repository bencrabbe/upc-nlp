
"""
These are artificial dataset generators for reasoning problems
"""

import random
import pandas as pa

from constraint import *
from itertools import product,permutations,combinations


nationalities = {"norwegian","spanish","japanese","english","ukrainian"}
colors        = {"yellow","blue","red","ivory","green"}
pet           = {"dog", "snails", "fox", "horse", "zebra"}
cigarette     = {"Old Gold", "Kools", "Chesterfields", "Lucky Strike", "Parliaments"}
beverage      = {"coffee", "milk", "orange juice", "water", "tea"}

class UnaryConstraint:

      def __init__(self,houseID,predicate):

          self.id        = houseID
          self.predicate = predicate

      def sample_text(self):

          ordinals = ['1st', '2nd', '3rd', '4th', '5th']
          noun     = random.choice(['guy','person','man'])

          if self.predicate in nationalities:
              return f"The {noun} in the {ordinals[self.id-1]} house is {self.predicate}"
          elif self.predicate in colors:
              return f"The {ordinals[self.id-1]} house is {self.predicate}"
          elif self.predicate in pet:
              predicate = f"a {self.predicate}" if self.predicate != 'snails' else self.predicate
              return f"The {noun} in the {ordinals[self.id-1]} house owns a {predicate}"
          elif self.predicate in cigarette:
              return f"The {noun} living in the {ordinals[self.id-1]} house smokes {self.predicate}"
          elif self.predicate in beverage:
              return f"The {noun} living in the {ordinals[self.id-1]} house drinks {self.predicate}"
          else:
              raise NotImplementedError


      @staticmethod
      def generate_constraints(assignation_list):
        """
        Generates unary constraints given an assignation of values to their respective houses.
        """
        return [UnaryConstraint(idx+1,val) for idx,val in enumerate(assignation_list)]



class EqualityConstraint:

      def __init__(self,valueA,valueB):

          self.valA  = valueA
          self.valB  = valueB

          if self.valA in colors:
             self.valA,self.valB = self.valB,self.valA


      def sample_text(self):

          if self.valA in nationalities:
            if self.valB in colors:
               return f"The {self.valA} lives in the {self.valB} house"
            elif self.valB in pet:
              feature = f"a {self.valB}" if self.valB != 'snails' else self.valB
              return f"The {self.valA}  owns {feature}"
            elif self.valB in cigarette:
              return f"The {self.valA}  smokes {self.valB}"
            elif self.valB in beverage:
              return f"The {self.valA}  drinks {self.valB}"

          elif self.valA in pet:
            if self.valB in nationalities:
              if self.valA == "snails":
                return f"The snails live with the {self.valB}"
              else:
                return f"The {self.valA}  is owned by the {self.valB}"
            elif self.valB in cigarette:
              return f"The {self.valA} lives with the guy who smokes {self.valB}"
            elif self.valB in colors:
               return f"The {self.valA} lives in the {self.valB} house"
            elif self.valB in beverage:
              return f"The {self.valA} lives with the  {self.valB} drinker"

          elif self.valA in beverage:

              if self.valB in nationalities:
                return f"The {self.valA}  drinker lives in the {self.valB} house"
              elif self.valB in colors:
                return f"The {self.valA} drinker lives in the {self.valB} house"
              elif self.valB in pet:
                return f"The {self.valA}  drinker owns a {self.valB}"
              elif self.valB in cigarette:
                return f"The {self.valA}  drinker smokes {self.valB}"

          elif self.valA in cigarette:
              if self.valB in nationalities:
                return f"The {self.valA} smoker is {self.valB}"
              elif self.valB in colors:
                return f"The {self.valA} smoker lives in the {self.valB} house"
              elif self.valB in pet:
                return f"The {self.valA} smoker owns a {self.valB}"
              elif self.valB in beverage:
                return f"The {self.valA} smoker drinks {self.valB}"



          raise NotImplementedError(f"A : {self.valA}, B: {self.valB}")


      @staticmethod
      def generate_constraints(assignation_listA,assignation_listB):
        """
        Generates equality constraints given an assignation of values to their respective houses.
        """
        return [EqualityConstraint(valA,valB) for valA,valB in zip(assignation_listA,assignation_listB)]



class ProblemGenerator:

    @staticmethod
    def generate_problem(nationalities,pets,cigarettes,colors,beverages):

        assignations_dict = {'nationalities':nationalities,
                             'pet':pets,
                             'cigarette':cigarettes,
                             'colors':colors,
                             'beverage':beverages}

        #generate constraints
        clist = []
        for family in assignations_dict.values():
            clist.extend(UnaryConstraint.generate_constraints(family))

        combos = combinations(['nationalities','pet','cigarette','colors','beverage'],2)
        for keyA,keyB in combos: 
            clist.extend(EqualityConstraint.generate_constraints(assignations_dict[keyA],assignations_dict[keyB]))

        random.shuffle(clist)

        sols = []
        last = None
        while len(sols) <= 1:
          last = clist.pop()
          P = ProblemGenerator.encode_problem(assignations_dict,clist)
          sols = P.getSolutions()

        if last:
          clist.append(last)
        return ( pa.DataFrame(assignations_dict), ProblemGenerator.to_dict(assignations_dict,clist) )



    @staticmethod
    def sample_dataset(nhouses,full_size,seed=7):
        
        random.seed(seed)

        all_assignations = []
        for k in range(nhouses):

            vardoms = {'nationalities':random.sample(list(nationalities),k),
                     'pet':random.sample(list(pet),k),
                     'cigarette':random.sample(list(cigarette),k),
                     'colors':random.sample(list(colors),k),
                     'beverage':random.sample(list(beverage),k)}
            
            varassignations = {'nationalities':permutations(vardoms['nationalities']),
                               'pet':permutations(vardoms['pet']),
                               'cigarette':permutations(vardoms['cigarette']),
                               'colors':permutations(vardoms['colors']),
                               'beverage':permutations(vardoms['beverage'])}
        
            all_assignations.extend(product(varassignations['nationalities'],
                                            varassignations['pet'],
                                            varassignations['cigarette'],
                                            varassignations['colors'],
                                            varassignations['beverage']))

        print('Theoretical number of puzzles',len(all_assignations))
        random.shuffle(all_assignations)     


        all_data = []
        for assignation in all_assignations:
            solution, data_item = ProblemGenerator.generate_problem(*assignation)
            item_len = sum(len(elt['content'].split()) for elt in data_item)
            if item_len < 350: #drops unusually long examples where inference can be broken
                all_data.append(data_item)        
            if len(all_data) >= full_size:
               break        

        all_data = ProblemGenerator.flatten_problems(all_data)        
        
        splitA,splitB = int(0.8*len(all_data)),int(0.9*len(all_data))
        random.shuffle(all_data)
        train  = all_data[:splitA]
        valid  = all_data[splitA:splitB]
        test   = all_data[splitB:]

        print(f"Generated full dataset of {len(all_data)} examples")
        print(f"Train set size {len(train)}, Valid set size {len(valid)}, Test set size {len(test)}")
        return (train,valid,test)


    @staticmethod
    def flatten_problems(dictlist):
        """
        This flattens the dataset in a standardized message format easier to use with the huggingface interface 
        """
        flat_data = []

        for zebra_problem in dictlist:
                 
            system_instruction,task_context,constraints,*QA = zebra_problem    
            for idx in range(0,len(QA),2):
              question = QA[idx]
              answer   = QA[idx+1]

              user       = '\n'.join([ system_instruction['content'], task_context['content'],constraints['content'],question['content'] ])
              assistant  = answer['content']

              flat_data.append({"messages": [{"role": "user", "content": user},
                                             {"role": "assistant", "content": assistant} ]})
        return flat_data


    @staticmethod
    def sample_questions(pdict):
        """
        Samples one question for each criterion in pdict.
        """

        chosen_pet = random.choice(pdict['pet'])
        pet_idx    = pdict['pet'].index(chosen_pet)

        chosen_beverage = random.choice(pdict['beverage'])
        beverage_idx    = pdict['beverage'].index(chosen_beverage)

        chosen_cigarette = random.choice(pdict['cigarette'])
        cigarette_idx    = pdict['cigarette'].index(chosen_cigarette)

        chosen_color = random.choice(pdict['colors'])
        color_idx    = pdict['colors'].index(chosen_color)


        qa = [(f"who owns the {chosen_pet} ?", f"the {pdict['nationalities'][pet_idx]}"),
              (f"who smokes {chosen_cigarette} ?", f"the {pdict['nationalities'][cigarette_idx]}"),
              (f"who drinks {chosen_beverage} ?", f"the {pdict['nationalities'][beverage_idx]}"),
              (f"who lives in the {chosen_color} house ?", f"the {pdict['nationalities'][color_idx]}")]

        return qa

    @staticmethod
    def to_dict(assignations_dict,constraints):
        """
        The returned format is compatible with the open AI chatml formatting
        """
        def enum_an_string(values):
          return f"an {', an '.join(values[:-1])} and an {values[-1]}"

        def enum_string(values):
          return f"{', '.join(values[:-1])} and {values[-1]}"

        def enum_the_string(values):
          return f"the {', the '.join(values[:-1])} and the {values[-1]}"

        nhouses = len(assignations_dict['nationalities'])

        description = '\n'.join([f"There are {nhouses} houses.",
                       f"There is {enum_an_string(assignations_dict['nationalities'])}",
                       f"there are {enum_string(assignations_dict['colors'])} houses",
                       f"the pets are {enum_the_string(assignations_dict['pet'])}",
                       f"the persons smoke {enum_string(assignations_dict['cigarette'])}",
                       f"and they drink {enum_string(assignations_dict['beverage'])}."])

        constraints = '\n'.join([c.sample_text() for c in constraints])

        msg_lst = [{"role":"system","content":"We want a very short answer to the following riddle. "},
                   {"role":"user","content":description},
                   {"role":"user","content":constraints}]

        for Q,A in ProblemGenerator.sample_questions(assignations_dict):
                    msg_lst.append( {"role":"user","content":Q} )
                    msg_lst.append( {"role":"assistant","content":A} )
        return msg_lst



    @staticmethod
    def encode_problem(assignations_dict,constraintList):

        problem = Problem()
        nhouses = len(assignations_dict['nationalities'])
        criteria = [item for predicates in assignations_dict.values() for item in predicates]

        problem.addVariables(criteria,[1,2,3,4,5][:nhouses])
        problem.addConstraint(AllDifferentConstraint(), assignations_dict['nationalities'])
        problem.addConstraint(AllDifferentConstraint(), assignations_dict['pet'])
        problem.addConstraint(AllDifferentConstraint(), assignations_dict['cigarette'])
        problem.addConstraint(AllDifferentConstraint(), assignations_dict['colors'])
        problem.addConstraint(AllDifferentConstraint(), assignations_dict['beverage'])

        for c in constraintList:
            if type(c) == UnaryConstraint:
              problem.addConstraint(InSetConstraint([c.id]),[c.predicate])

            elif type(c) == EqualityConstraint:
              problem.addConstraint(lambda x,y: x == y, [c.valA,c.valB])

        return problem




if __name__ == "__main__":



    import argparse
    import json
    import pprint

    parser = argparse.ArgumentParser(
                    prog='Reasoning dataset generator',
                    description='Generates artificial data sets')

    parser.add_argument('fileprefix')   
    parser.add_argument('--nationalities',default=5,type=int)
    parser.add_argument('--max_size',default=200,type=int)
    args = parser.parse_args()
    if args.nationalities > 5:
       args.nationalities = 5
       print('there can not be more than 5 nationalities')
    if args.nationalities < 2:
       args.nationalities = 2
       print('there can be no less than 2 nationalities')

    train,valid,test = ProblemGenerator.sample_dataset(args.nationalities,args.max_size)

    with open(f"{args.fileprefix}-train.json",'w') as trainf:
        trainf.write(json.dumps(train))
    with open(f"{args.fileprefix}-valid.json",'w') as validf:
        validf.write(json.dumps(valid))
    with open(f"{args.fileprefix}-test.json",'w') as testf:
        testf.write(json.dumps(test))

    print("*** Example data item ***")
    pprint.pprint(train[0])

