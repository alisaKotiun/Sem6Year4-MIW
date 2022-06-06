/* Needed by SWI-Prolog. */
:- dynamic passed/4, not_passed/3, at_position/2, current_position/1.   
:- retractall(at_position(_, _)), retractall(current_position(_)), retractall(not_passed(_)), retractall(passed(_)).


/* Initializing current position */
current_position(main_square).

/* FACTs for passing tests */
not_passed(test1).
not_passed(test2).


/* RULEs for connections between places */
path(main_square, front, building_b).
path(building_b, back, main_square).

path(main_square, back, building_a).
path(building_a, front, main_square).

path(main_square, right, smoking_area).
path(smoking_area, left, main_square).

path(building_b, front, dean_office) :- passed(test1), passed(test2), !.
path(building_b, front, dean_office) :-
    write('  ~ hold on, you can*t go to the dean*s office until '), nl,
    write('all the assignments are done!'), nl, 
    position, nl, !, fail.


/* FACTs for objects's location */
at_position(bench, smoking_area).
at_position(some_student, smoking_area).
at_position(sushi, building_a).
at_position(onigiri, building_a).
at_position(coffee, building_a).
at_position(dr_werner, bench).


/* RULEs for directions */
front :- go(front). 
back :- go(back).
left :- go(left). 
right :- go(right).


/* RULEs for moving */
go(Direction) :-
    current_position(Current),
    path(Current, Direction, Following),
    retract(current_position(Current)),
    assert(current_position(Following)),
    position, !.

go(_) :-
    write('  ~ oops.. no way').


/* RULE: where you are */
position :-
    current_position(Current),
    describe(Current), nl, 
    show_objects(Current), 
    nl.


/* RULEs for describing objects */
show_objects(Location) :-
    at_position(X, Location),
    write('Hey, there is a '), write(X), write(' in '), write(Location), nl,
    show_objects(X),
    fail.

show_objects(_).


/*RULEs for taking a test1 */
amtest :-
    not_passed(test1),
    current_position(building_b),
    write(' Calculate the following expression:'), nl,
    write('2 + 3 ='), nl, nl, 
    write('Enter result with amtest_result(*number*)'),
    nl, !.

amtest :-
    current_position(building_b),
    write('All tests are already written.'), nl,
    write('Move on!'),
    nl, !.

amtest :- 
    write('There is no AM test to write'), nl.

amtest_result(Result) :-
    Result =:= 5,
    retract(not_passed(test1)),
    assert(passed(test1)),
    write('    !!!Correct answer!!!'), nl,
    write('Move forward!'), nl, nl,
    position,
    nl, !.

amtest_result(Result) :-
    write('    ...Wrong answer...'), nl.


/*RULEs for taking a test2 */
werner_test(List) :-
    member(List, [public, private, protected, default]),
    retract(not_passed(test2)),
    assert(passed(test2)),
    write('    !!!Correct answer!!!'), nl,
    write('Move forward!'), nl, nl,
    position,
    nl, !.

werner_test(List) :-
    write('    ...Wrong answer...'), nl.


/*RULEs for eating */
eat(Food) :-
    at_position(Food, building_a),
    retract(at_position(Food, building_a)),
    write('Mmmmm... so tasty!'), nl, nl,
    position,
    nl, !.

eat(Food) :-
    write('    ...There is no such a food'), nl.


/* RULE: how to go home */
go_home :-
    !, finish.


finish :-
    nl,
    write('    *** THE GAME IS OVER ***'), nl,
    write('Please enter the -halt.- command'),
    nl, !.


/* RULE: instructions */
instructions :-
    nl,
    write('       *** INSTRUCTIONS ***'), nl,
    write('  You need to use the following commands:'), nl,
    write('start.               ~ start the game'), nl,
    write('position.            ~ find out your position'), nl,
    write('front. back.         ~ go in the according'), nl,
    write('left. right.           direction'), nl,
    write('instructions.        ~ show instructions'), nl,
    write('go_home.             ~ go home and finish the game.'), nl, nl, nl,
    
    write('To win you need to complete all the assignments (2) to go home.'), nl,
    write('             GOOD LUCK!'), nl, nl,
    nl.


/* RULE: instructions + gamer's position */
start :-
    write('   *** WELCOME TO THE GAME ***'),
    instructions,
    position.


/* RULEs for descriptions */
describe(main_square) :-
    write('You are on the "Main Square" of PJWSTK. You are looking'), nl,
    write('at the "Building B", while a "Building A" is behind you.'), nl,
    write('A "Smoking Area" is on the right.'), nl,
    nl.

describe(building_a) :-
    at_position(X, building_a),
    write('You are in the "Building A"'), nl,
    write('There is a "Main Square" in the front'), nl,
    write('    ~Are you hungry? Click eat(*something*).'), nl, !.

describe(building_a) :-
    write('You are in the "Building A"'), nl,
    write('There is a "Main Square" in the front'), nl,
    nl.

describe(building_b) :-
    not_passed(test1),
    write('You are in the "Building B"'), nl, nl,
    write('Seems like you didn*t pass the AM test...'), nl,
    write('If you want to go home, write a test (amtest.)'), nl, !.

describe(building_b) :-
    write('You are in the "Building B"'), nl,
    write(' There is only a dean office in front of you'), nl,
    write('and a "Main Square" in the back'), nl,
    write('Would you like to go in?'),
    nl.

describe(smoking_area) :-
    not_passed(test2),
    write('You are in the "Smoking Area"'), nl, 
    write('Seems like Dr Werner wants to ask you a question:'), nl,
    write('    - Good morning! I can*t remind myself what '), nl,
    write('    are the access modifiers in Java.'), nl,
    write('    Name at least 1, please.'), nl,
    write('Call -werner_test(answer).- to answer Dr Werner*s question.'), nl, !.

describe(smoking_area) :-
    write('You are in the "Smoking Area"'), nl, 
    write('You can turn left to return to "Main Square"'), 
    nl.

describe(dean_office) :-
    write('You are in the "Dean*s Office"'), nl,
    write(' and all assignments are completed!'), nl,
    write('Now you can go home!'), nl, nl, nl,
    go_home.
