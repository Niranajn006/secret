#%% Test
# from pony import orm
from pony.orm import *

#%% Initial create database
db = Database()

# The classes that we have created are derived from the Database.
# Entity attribute of the Database object. It means that they are not ordinary classes, but entities.
# The entity instances are stored in the database, which is bound to the db variable

class Person(db.Entity):
    name = Required(str) # Required mean Not None value
    age = Required(int)
    cars = Set('Car') # Set keyword is a relationship. It can keep a collection of instances of the "Car" entity

class Car(db.Entity):
    make = Required(str)
    model = Required(str)
    owner = Required(Person)
#The "Car" entity has three mandatory attributes: make and model are strings,
#  and the owner attribute is the other side of the one-to-many relationship. 

#**If we need to create a many-to-many relationship between two entities, we should declare two Set attributes at both ends

#%% Check Entitiy that we have craete
show(Person)

#%% Database binding
db.bind(provider='sqlite', filename=':memory:')

#%% Mapping entities to database tables
db.generate_mapping(create_tables=True) #if the tables do not already exist, then they will be created using the CREATE TABLE command.

#%% Set debug mode
set_sql_debug(True)

#%% Create entity instances
p1 = Person(name='John', age=20)
p2 = Person(name='Mary', age=22)
p3 = Person(name='Bob', age=30)
c1 = Car(make='Toyota', model='Prius', owner=p2)
c2 = Car(make='Ford', model='Explorer', owner=p3)
commit() #Pony does not save objects in the database immediately. These objects will be saved only after the commit()

#%% db_session 
#But when you use Pony in your application,
# all database interactions should be done within a database session.
@db_session
def print_person_name(person_id):
    p = Person[person_id]
    print(p.name)

@db_session
def add_car(person_id, make, model):
    Car(make=make, model=model, owner=Person[person_id])

#Another option for working with the database is using the db_session()
#  as the context manager instead of the decorator:

with db_session:
    p = Person(name='Kate', age=33)
    Car(make='Audi', model='R8', owner=p)

#%% Writing queries
select(p for p in Person if p.age > 20)[:]
# One of the ways to get the list of objects is to apply the slice operator [:] 

#%% Writing and show data
select(p for p in Person).order_by(Person.name)[:2].show()

#%% Queries with iterate over the resulting 
persons = select(p for p in Person if 'o' in p.name)
for p in persons:
    print(p.name, p.age)

#%% queries object attribute
select(p.name for p in Person if p.age != 30)[:]


#%% Or list of tuple 
select((p, count(p.cars)) for p in Person)[:]


#%% With Pony you can also run aggregate queries. 
# Here is an example of a query which returns the maximum age of a person:
max(p.age for p in Person)

#%% Getting objects by its primary key need specify the primary key value
p1 = Person[1]
print(p1.name)


#%% For retrieving the objects by other attributes, you can use the Entity.get() method:
mary = Person.get(name='Mary')
print(mary.age)


#%% Writing raw SQL queries
x = 25 # declare variable 
Person.select_by_sql('SELECT * FROM Person p WHERE p.age < $x')

#%% more Ponyorm example database
new_db = Database()
new_db.bind(provider='sqlite', filename='C:\\sqlite\\test.db', create_db=True)


#%%
new_db.generate_mapping(create_tables=True)


#%%
data = new_db.select("select * from tbl1")
data

#%%
