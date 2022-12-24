"""
Simulator.py, mostly just https://github.com/OptimusLime/iesor-physics
translated to python

Deprecated file. See simulator.py.
"""

import json
import math
import os

from Box2D import Box2D as b2
from Box2D.examples.framework import Framework, main

from openelm.environments.sodaracer.helpers import (
    BodyInformation,
    Bone,
    DistanceAccessor,
    Entity,
    EntityType,
    Muscle,
    PhysicsId,
    staticBodyCount,
)


class IESoRWorld(Framework):
    def __init__(self):
        super(IESoRWorld, self).__init__()
        self.interpolation = 0.0
        self.radians = 0.0
        self.accumulator = 0.0
        self.desiredFPS = 0.0
        self.simulation_rate = 0.0
        # keep a list of body identifiers
        self.body_list = list()
        # keep a list of shape identifiers
        self.shape_list = list()
        # keep a reference to every body according to ID
        self.body_map = dict()
        # keep track of all the joints inside the system
        self.bone_list = list()
        # we need to keep track of all our muscles too
        self.muscle_list = list()

        #
        #
        # IESoRWorld::IESoRWorld()
        # {
        #     this->interpolation =0;
        self.interpolation = 0
        #     accumulator = 0;
        self.accumulator = 0
        #     desiredFPS = 60;
        self.desiredFPS = 60
        #     simulationRate = 1.0 / desiredFPS;
        self.simulation_rate = 1.0 / self.desiredFPS
        #     radians = 0;
        self.radians = 0
        #
        #
        #     //this->bodyList = new vector<PhysicsID*>();
        #     //this->shapeList = new vector<PhysicsID*>();
        #     //this->bodyList = new Json::Value(Json::arrayValue);
        #     //this->shapeList = new Json::Value(Json::arrayValue);
        #
        #     // Define the gravity vector.
        #     b2Vec2 gravity(0.0f, -15.0f);
        self.gravity = b2.b2Vec2(0.0, -15.0)  # ??? Magic numbers.
        #
        #     // Construct a world object, which will hold and simulate the rigid bodies.
        #     this->world = new b2World(gravity);
        #     this->world->SetAutoClearForces(false);
        # self.world = b2.b2World(self.gravity)
        self.world.gravity = self.gravity
        self.world.autoClearForces = False
        #
        #     // Define the ground body.
        #     b2BodyDef groundBodyDef;
        #     groundBodyDef.type = b2_staticBody;
        #     groundBodyDef.position.Set(0.0f, -18.0f);
        self.groundBodyDef = b2.b2BodyDef()
        self.groundBodyDef.type = b2.b2_staticBody
        self.groundBodyDef.position = b2.b2Vec2(0.0, -18.0)
        #
        #     // Call the body factory which allocates memory for the ground body
        #     // from a pool and creates the ground box shape (also from a pool).
        #     // The body is also added to the world.
        #     b2Body* groundBody = this->addBodyToWorld(
        #         "ground", &groundBodyDef);//this->world->CreateBody(&groundBodyDef);
        self.groundBody = self.add_body_to_world("ground", self.groundBodyDef)
        #     //b2Body* groundBody = this->world->CreateBody(&groundBodyDef);
        #
        #
        #     // Define the ground box shape.
        #     b2PolygonShape groundBox;
        self.groundBox = b2.b2PolygonShape()
        #
        #     // The extents are the half-widths of the box.
        #     groundBox.SetAsBox(350.0, 10.0);
        self.groundBox.SetAsBox(350.0, 10.0)
        #
        #     // Add the ground fixture to the ground body.
        #     this->addShapeToBody(groundBody, &groundBox, 0.0f);
        self.add_shape_to_body(self.groundBody, self.groundBox, 0.0)
        #     //groundBody->CreateFixture(&groundBox, 0.0f);
        #
        #     //Pulling in some sample data for testing!
        #     std::string bodyJSON = loadDataFile("sampleBody224632.json");
        #
        #     //pull in our JSON body plz
        #     Json::Value inBody;
        #     Json::Reader read;
        #     read.parse(bodyJSON, inBody, true);
        self.body_json = self.load_data_file("jsBody142856.json")
        self.in_body = json.loads(self.body_json)
        #
        #     b2Vec2 canvas(200, 150);
        self.canvas = b2.b2Vec2(200, 150)
        #     this->loadBodyIntoWorld(inBody, canvas);
        self.load_body_into_world(self.in_body, self.canvas)
        #
        #
        # #     // Define the dynamic body. We set its position and call the body factory.
        # #     b2BodyDef bodyDef;
        # self.body_def = b2.b2BodyDef()
        # #     bodyDef.type = b2_dynamicBody;
        # self.body_def.type = b2.b2_dynamicBody
        # #     bodyDef.position.Set(0.0f, 24.0f);
        # self.body_def.position = b2.b2Vec2(0.0, 24.0)
        # #
        # #     //add body to world using definition
        # #     b2Body* body = this->addBodyToWorld("rect1", &bodyDef);
        # self.body = self.add_body_to_world("rect1", self.body_def)
        # #
        # #     // Define another box shape for our dynamic body.
        # #     b2PolygonShape dynamicBox;
        # self.dynamic_box = b2.b2PolygonShape()
        # #     dynamicBox.SetAsBox(5.0f, 5.0f);
        # self.dynamic_box.SetAsBox(5.0, 5.0)
        # #
        # #     // Define the dynamic body fixture.
        # #     b2FixtureDef fixtureDef;
        # self.fixture_def = b2.b2FixtureDef()
        # #     fixtureDef.shape = &dynamicBox;
        # self.fixture_def.shape = self.dynamic_box
        # #     fixtureDef.restitution = .5;
        # self.fixture_def.restitution = .5
        # #
        # #     // Set the box density to be non-zero, so it will be dynamic.
        # #     fixtureDef.density = 1.0f;
        # self.fixture_def.density = 1.0
        # #
        # #     // Override the default friction.
        # #     fixtureDef.friction = 0.3f;
        # self.fixture_def.friction = 0.3
        # #
        # #     // Add the shape to the body.
        # #     this->addShapeToBody(body, &fixtureDef);
        # self.add_shape_to_body_fixture(self.body, self.fixture_def)
        # #
        # #     // Define the circle body. We set its position and call the body factory.
        # #     b2BodyDef cDef;
        # self.c_def = b2.b2BodyDef()
        # #     cDef.type = b2_dynamicBody;
        # self.c_def.type = b2.b2_dynamicBody
        # #     cDef.position.Set(10.0f, 24.0f);
        # self.c_def.position = b2.b2Vec2(10.0, 24.0)
        # #
        # #     //add body to world using definition
        # #     body = this->addBodyToWorld("circleTest", &cDef);
        # self.body = self.add_body_to_world("circleTest", self.c_def)
        # #
        # #     // Define another box shape for our dynamic body.
        # #     b2CircleShape dCircle;
        # self.d_circle = b2.b2CircleShape()
        # #     dCircle.m_radius = 5.0;
        # self.d_circle.radius = 5.0
        # #
        # #     // Define the dynamic body fixture.
        # #     b2FixtureDef circleDef;
        # self.circle_def = b2.b2FixtureDef()
        # #     circleDef.shape = &dCircle;
        # self.circle_def.shape = self.d_circle
        # #     circleDef.restitution = .5;
        # self.circle_def.restitution = 0.5
        # #
        # #     // Set the box density to be non-zero, so it will be dynamic.
        # #     circleDef.density = 1.0f;
        # self.circle_def.density = 1.0
        # #
        # #     // Override the default friction.
        # #     circleDef.friction = 0.3f;
        # self.circle_def.friction = 0.3
        # #
        # #     // Add the shape to the body.
        # #     this->addShapeToBody(body, &circleDef);
        # # self.add_shape_to_body_fixture(self.body, self.circle_def)
        # #
        # #
        # #
        # #     //Add some forces!
        # #     //body->ApplyAngularImpulse(.4, true);
        # #
        # #     body->ApplyTorque(150, true);
        # self.body.ApplyTorque(150, True)
        # #     b2Vec2 pulse(70, 0);
        # #     b2Vec2 o(0,3);
        # #     body->ApplyLinearImpulse(pulse, o, true);
        # self.body.ApplyLinearImpulse(
        #     b2.b2Vec2(70, 0), b2.b2Vec2(0, 3), True
        # )
        # #
        # # }

    def worldDrawList(self) -> str:
        # string IESoRWorld::worldDrawList()
        # {
        #     //This will be written to json
        #     //fastwriter is not human readable --> compact
        #     //exactly what we want to sent to our html files
        #     Json::FastWriter* writer = new Json::FastWriter();
        #
        #     //root is what we'll be sending back with all the shapes (stored in shapes!)
        #     Json::Value root;
        root = dict()
        #     Json::Value shapes;
        shapes = dict()
        #
        #     //we'll loop through all required body information
        #     b2Body * B = this->world->GetBodyList();
        b_list = self.world.bodies
        #
        #     while(B != NULL)
        for b in b_list:  # type: b2.b2Body
            userData = b.userData
            if userData is not None:
                #     {
                #         if(B->GetUserData())
                #         {
                #             //fetch our body identifier
                #             PhysicsID* pid = static_cast<PhysicsID*>(B->GetUserData());//*((string*)B->GetUserData());
                pid = userData  # type: PhysicsId
                #             std::string bodyID = pid->ID();
                body_id = pid.id()

                #
                #             //we must get all our shapes
                #             b2Fixture* F = B->GetFixtureList();
                f_list = b.fixtures
                #
                #             //cycle through the shapes
                #             while(F != NULL)
                #             {
                for f in f_list:  # type: b2.b2Fixture
                    #                 //Hold our shape drawing information
                    #                 Json::Value singleShape;
                    single_shape = dict()
                    #
                    #                 switch (F->GetType())
                    #                 {
                    #                 case b2Shape::e_circle:
                    if f.type == b2.b2Shape.e_circle:
                        #                     {
                        #                         b2CircleShape* circle = (b2CircleShape*) F->GetShape();
                        circle = f.shape  # type: b2.b2CircleShape
                        #                         /* Do stuff with a circle shape */
                        #                         Json::Value center = positionToJSONValue(circle->m_p);
                        center = position_to_json_value(circle.pos)
                        #                         Json::Value radius = circle->m_radius;
                        radius = circle.radius
                        #                         singleShape["type"] = "Circle";
                        single_shape["type"] = "circle"
                        #                         singleShape["bodyOffset"] = positionToJSONValue(B->GetPosition());
                        single_shape["bodyOffset"] = position_to_json_value(b.position)
                        #                         singleShape["rotation"] = B->GetAngle();
                        single_shape["rotation"] = b.angle
                        #                         singleShape["center"] = center;
                        single_shape["center"] = center
                        #                         singleShape["radius"] = circle->m_radius;
                        single_shape["radius"] = radius
                        #                         singleShape["color"] = "#369";
                        single_shape["color"] = "#369"
                    #                     }
                    #                     break;
                    elif f.type == b2.b2Shape.e_polygon:
                        #                 case b2Shape::e_polygon:
                        #                     {
                        #                         b2PolygonShape* poly = (b2PolygonShape*) F->GetShape();
                        poly = f.shape  # type: b2.b2PolygonShape
                        #                         /* Do stuff with a polygon shape */
                        #
                        #                         Json::Value points = listOfPoints(poly->m_vertices, poly->m_count);
                        #                         singleShape["type"] = "Polygon";
                        single_shape["type"] = "polygon"
                        #                         singleShape["points"] = points;
                        single_shape["points"] = list_of_points(
                            poly.vertices, poly.vertexCount
                        )
                        #                         singleShape["bodyOffset"] = positionToJSONValue(B->GetPosition());
                        single_shape["bodyOffset"] = position_to_json_value(b.position)
                        #                         singleShape["rotation"] = B->GetAngle();
                        single_shape["rotation"] = b.angle
                        #                         singleShape["color"] = "#38F";
                        single_shape["color"] = "#38F"
                    #
                    #                     }
                    #                     break;
                    #                 }
                    #
                    #                 //Each shape is the unique combination of
                    #                 pid = static_cast<PhysicsID*>(F->GetUserData());//*((string*)B->GetUserData());
                    pid = f.userData
                    #
                    #                 string shapeID = pid->ID();// *((string*)F->GetUserData());
                    shape_id = pid.id()
                    #
                    #                 string fullID = bodyID + "_" + shapeID;
                    full_id = body_id + shape_id
                    #                 shapes[fullID] = singleShape;
                    shapes[full_id] = single_shape
        #
        #                 F = F->GetNext();
        #             }
        #         }
        #
        #         B = B->GetNext();
        #     }
        #     //we set our shapes using the body loops
        #     root["shapes"] = shapes;
        root["shapes"] = shapes
        #
        #     //we now need to process all of our joints as well
        #     b2Joint * J = this->world->GetJointList();
        j_list = self.world.joints
        #
        #     Json::Value joints;
        joints = dict()
        #
        #     while(J != NULL)
        for j in joints:  # type: b2.b2Joint
            #     {
            #         if(J->GetUserData())
            if j.userData is not None:
                #         {
                #             //fetch our body identifier
                #             Bone* jid = static_cast<Bone*>(J->GetUserData());
                jid = j.userData  # type: Bone
                #
                #             //we grab the joint identifier
                #             std::string bodyID = jid->ID();
                body_id = jid.id()
                #
                #             //Hold our joint drawing information
                #             Json::Value singleJoint;
                single_joint = dict()
                #
                #             //we should use the body identifiers
                #             //but they both need to exist for this to be valid
                #             if(J->GetBodyA()->GetUserData() && J->GetBodyB()->GetUserData())
                if (j.bodyA.userData is not None) and (j.bodyB.userData is not None):
                    #             {
                    #                 //we need to know what bodies are connected
                    #                 PhysicsID* pid = static_cast<PhysicsID*>(J->GetBodyA()->GetUserData());
                    pid = j.bodyA.userData  # type: PhysicsId
                    #                 singleJoint["sourceID"] =  pid->ID();
                    single_joint["sourceID"] = pid.id()
                    #
                    #                 pid = static_cast<PhysicsID*>( J->GetBodyB()->GetUserData());
                    pid = j.bodyB.userData  # type: PhysicsId
                    #                 singleJoint["targetID"] =  pid->ID();
                    single_joint["targetID"] = pid
                    #
                    #                 //now we need more drawing informtion regarding offsets on the body
                    #
                    #                 //get the anchor relative to body A
                    #                 //set that in our json object
                    #                 singleJoint["sourceOffset"] = positionToJSONValue(J->GetAnchorA());
                    single_joint["sourceOffset"] = position_to_json_value(j.anchorA)
                    #                 //set the same for our second object
                    #                 singleJoint["targetOffset"] = positionToJSONValue(J->GetAnchorB());
                    single_joint["targetOffset"] = position_to_json_value(j.anchorB)
                #             }
                #
                #             //set our joint object using the id, and json values
                #             joints[bodyID] = singleJoint;
                joints[body_id] = single_joint
        #         }
        #
        #         //loop to the next object
        #         J = J->GetNext();
        #     }
        #
        #     root["joints"] = joints;
        root["joints"] = joints
        #
        #     return writer->write(root);
        # }
        return json.dumps(root)

    def Step(self, settings):
        """
        The main physics step.

        Takes care of physics drawing (callbacks are executed after the world.Step() )
        and drawing additional information.
        """

        # self.stepCount += 1
        # # Don't do anything if the setting's Hz are <= 0
        if settings.hz > 0.0:
            timeStep = 1.0 / settings.hz
        else:
            timeStep = 0.0
        self.updateWorld(timeStep * 1000.0)
        settings.velocityIterations = 10
        settings.positionIterations = 10
        settings.hz = 0.0
        super().Step(settings)
        self.world.ClearForces()

    def updateWorld(self, ms_update: float) -> int:
        #
        #
        # //pass in the amount of time you want to simulate
        # //in real time, this is the delta time between called to update
        # //in simulation time, this can be any amount -- but this function won't return until all updates are processed
        # //so it holds up the thread
        # int IESoRWorld::updateWorld(double msUpdate)
        # {
        #     msUpdate = 4*msUpdate;
        ms_update = 4 * ms_update  # why...
        #
        #     //we'll keep track of the number of times we simulated during this update
        #     int stepCount = 0;
        step_count: int = 0
        #
        #     //# of seconds since last time
        #     double frameTime = msUpdate/1000.0f;
        frame_time = ms_update / 1000.0
        #
        #     //maximum frame time, to prevent what is called the spiral of death
        #     if (frameTime > .35)
        #         frameTime = .35;
        frame_time = 0.35 if frame_time > 0.35 else frame_time
        #
        #     //we accumulate all the time we haven't rendered things in
        #     this->accumulator += frameTime;
        self.accumulator += frame_time
        #
        #     //as long as we have a chunk of time to simulate, we need to do the simulation
        #     while (accumulator >= simulationRate)
        #     {
        while self.accumulator >= self.simulation_rate:
            #         //how many times did we run this loop, we shoudl keep track --
            #         //that's the number of times we ran the physics engine
            #         stepCount++;
            step_count += 1
            #
            #         //move the muscles quicker using this toggle
            #         float speedup = 3;
            speedup = 3
            #
            #         //we loop through all our muscles, and update the lengths associated with the connectionsj
            #         for (std::vector<Muscle*>::iterator it = muscleList.begin() ; it != muscleList.end(); ++it)
            #         {
            for muscle in self.muscle_list:
                #             //grab our muscle pointer from the list
                #             Muscle* muscle = *it;
                #
                #             //get our distance joint -- holder of physical joint info
                #             b2DistanceJoint* dJoint = (b2DistanceJoint*)muscle->GetJoint();
                d_joint = muscle.get_joint()
                #
                #             //Javascript version
                #             //muscle.SetLength(muscle.m_length + muscle.amplitude/this.scale*Math.cos(rad + muscle.phase*2*Math.PI));
                #             //double lengthCalc = (dJoint->GetLength() + muscle->GetAmplitude()*cos(radians + muscle->GetPhase() * 2 * M_PI));
                #
                #             //fetch the original length of the distance joint, and add some fraction of that amount to the length
                #             //depending on the current location in the muscle cycle
                #             double lengthCalc = (1.0 + muscle->GetAmplitude() * cos(radians + muscle->GetPhase() * 2 * M_PI)) * muscle->GetOriginalLength();
                length_calc = (
                    1
                    + muscle.get_amplitude()
                    * math.cos(self.radians + muscle.get_phase() * 2 * math.pi)
                    * muscle.get_original_length()
                )
                #
                #             //we set our length as the calculate value
                #             dJoint->SetLength(lengthCalc);
                DistanceAccessor.setLength(d_joint, length_calc)
            #         }
            #
            #         //step the physics world
            #         this->world->Step(
            #             this->simulationRate   //frame-rate
            #             , 10       //velocity iterations
            #             , 10       //position iterations
            #         );
            self.world.Step(self.simulation_rate, 10, 10)
            #
            #         //manually clear forces when doing fixed time steps,
            #         //NOTE: that we disabled auto clear after step inside of the b2World
            #         this->world->ClearForces();
            self.world.ClearForces()
            #
            #         //increment the radians for the muscles
            #         radians += speedup * this->simulationRate;
            self.radians += speedup * self.simulation_rate
            #
            #         //decrement the accumulator - we ran a chunk just now!
            #         accumulator -= this->simulationRate;
            self.accumulator -= self.simulation_rate
        #     }
        #
        #     //interpolation is basically a measure of how far into the next step we should have simulated
        #     //it's meant to ease the drawing stages
        #     //(you linearly interpolate the last frame and the current frame using this value)
        #     this->interpolation = accumulator / this->simulationRate;
        self.interpolation = self.accumulator / self.simulation_rate
        #
        #     //how many times did we run the simulator?
        #     return stepCount;
        # }
        return step_count

    def add_body_to_world(self, body_id: str, body_def: b2.b2BodyDef) -> b2.b2Body:
        # b2Body* IESoRWorld::addBodyToWorld(string bodyID, b2BodyDef* bodyDef)
        # {
        #     //add body to world
        #     b2Body* body = this->world->CreateBody(bodyDef);
        body = self.world.CreateBody(body_def)  # type: b2.b2Body
        #
        #     //assign the bodyID to a new struct object
        #     PhysicsID* pid = new PhysicsID(bodyID);
        pid = PhysicsId(body_id)
        #
        #     //track all the struct objects
        #     this->bodyList.push_back(pid);
        self.body_list.append(pid)
        #
        #     //track all body objects
        #     this->bodyMap[bodyID] = body;
        self.body_map[body_id] = body
        #
        #     //identify body with user data
        #     body->SetUserData(pid);
        body.userData = pid
        #
        #     return body;
        # }
        return body

    def add_shape_to_body_fixture(
        self, body: b2.b2Body, fix_def: b2.b2FixtureDef
    ) -> b2.b2Fixture:
        # b2Fixture* IESoRWorld::addShapeToBody(b2Body* body, b2FixtureDef* fixDef)
        # {
        #     //create physics id based on global shape count
        #     PhysicsID* pid = createShapeID();
        #
        #     //add the fixture to our body using the definition
        #     b2Fixture* fix = body->CreateFixture(fixDef);
        #
        #     //Attach this info to the shape for identification
        #     setShapeID(fix, pid);
        #
        #     //send back created fixture
        #     return fix;
        # }
        pid = self.create_shape_id()
        fix = body.CreateFixture(fix_def)  # type: b2.b2Fixture
        set_shape_id(fix, pid)
        return fix

    def add_shape_to_body(
        self, body: b2.b2Body, shape: b2.b2Shape, density: float
    ) -> b2.b2Fixture:
        # b2Fixture* IESoRWorld::addShapeToBody(b2Body*body, b2Shape* shape, float density)
        # {
        #     //create physics id based on global shape count
        #     PhysicsID* pid = createShapeID();
        #
        #     //add the fixture to our body,
        #     b2Fixture* fix = body->CreateFixture(shape, density);
        #
        #     //Attach this info to the shape for identification
        #     setShapeID(fix, pid);
        #
        #     //send back created fixture
        #     return fix;
        # }
        pid = self.create_shape_id()
        fix = body.CreateFixture(shape=shape, density=density)  # type: b2.b2Fixture
        set_shape_id(fix, pid)
        return fix

    def load_body_into_world(self, in_body: dict, width_height: b2.b2Vec2) -> dict:
        #
        # std::map<std::string, double>* IESoRWorld::loadBodyIntoWorld(Json::Value& inBody, b2Vec2 widthHeight)
        # {
        # //we use bodycount as a way to identify each physical object being added from the body information
        # int oBodyCount = world->GetBodyCount();
        o_body_count = self.world.bodyCount
        #
        # //This mapping will be what we return from the function while adding the body
        # //we measure the morphological properties
        # map<string, double> morphology;
        morphology = dict()
        #
        # //here we use our body information to create all the necessary body additions to the world
        # vector<Json::Value> entities = getBodyEntities(inBody, widthHeight, morphology);
        entities = self.get_body_entities(in_body, width_height, morphology)
        #
        # //we'll quickly map and ID into a json point value
        # map<string, Json::Value> entityMap;
        entity_map = dict()
        #
        # //this was we can access locations by the bodyID while determining connection distance
        # for (std::vector<Json::Value>::iterator it = entities.begin() ; it != entities.end(); ++it)
        # {
        #     Json::Value e = *it;
        #     entityMap[e["bodyID"].asString()] = e;
        # }
        for e in entities:
            entity_map[e["bodyID"]] = e
        #
        # //push our bodies into the system so that our joints have bodies to connect to
        # this->setBodies(&entities);
        self.set_bodies(entities)
        #
        # //Did we use LEO to calculate our body information -- affects the indexing procedure
        # bool useLEO = inBody["useLEO"].asBool();
        useLEO = in_body["useLEO"]
        #
        # //this is the json connection array
        # Json::Value connections = inBody["connections"];
        connections = in_body["connections"]
        #
        # //this determines if we should be a fixed connection (bone) or a moving connection (muscle)
        # double amplitudeCutoff = .2;
        amplitude_cutoff = 0.2
        #
        # //we like to measure the total connection distance as part of our mass calculation
        # //mass = # of nodes + length of connections
        # double connectionDistanceSum = 0.0;
        connection_distance_sum = 0.0
        #
        # //loop through all our connections
        # for(int i=0; i < connections.size(); i++)
        for i in range(len(connections)):
            # {
            #     //grab connection from our array
            #     Json::Value connectionObject = connections[i];
            #     Json::Value cppnOutputs = connectionObject["cppnOutputs"];
            connection_object = connections[i]
            cppn_outputs = connection_object["cppnOutputs"]
            #
            #     int sID = atoi(connectionObject["sourceID"].asString().c_str());
            s_id = int(connection_object["sourceID"])
            t_id = int(connection_object["targetID"])
            #     int tID = atoi(connectionObject["targetID"].asString().c_str());
            #
            #     //To identify a given object in the physical world, we need to start with the current body count, and add the source id number
            #     //This allows us to add multiple bodies to the same world (though this is recommended against, since it's easier to create separate worlds)
            #     string sourceID = toString(oBodyCount + sID);
            #     string targetID = toString(oBodyCount + tID);
            source_id = str(o_body_count + s_id)
            target_id = str(o_body_count + t_id)
            #
            #      //we ignore connections that loop to themselves
            #     if (sourceID == targetID)
            #     {
            #         continue;
            #     }
            if source_id == target_id:
                continue
            #
            #
            #     try
            #     {
            try:  # oh boy...
                #         //depending on whether LEO was used or not, dictates what outputs we'll be looking at for what
                #         int phaseIx = (useLEO ? 2 : 1);
                #         int ampIx = (useLEO ? 3 : 2);
                phase_ix = 2 if useLEO else 1
                amp_ix = 3 if useLEO else 2
                #
                #         //sample the amplitude output to know what's up -- we convert from [-1,1] to [0,1]
                #         double amp = (cppnOutputs[ampIx].asDouble() + 1) / 2;
                amp = (cppn_outputs[amp_ix] + 1) / 2.0
                #
                #         //what's the distance been the source (x,y) and distance (x,y) -- that's the length of our connection
                #         double connectionDistance = sqrt(
                #             pow(entityMap[sourceID]["x"].asDouble() - entityMap[targetID]["x"].asDouble(), 2)
                #             + pow(entityMap[sourceID]["y"].asDouble() - entityMap[targetID]["y"].asDouble(), 2));
                #
                connection_distance = math.sqrt(
                    (entity_map[source_id]["x"] - entity_map[target_id]["x"]) ** 2
                    + (entity_map[source_id]["y"] - entity_map[target_id]["y"]) ** 2
                )
                #         //add the connection distance to our sum
                #         connectionDistanceSum += connectionDistance;
                connection_distance_sum += connection_distance
                #
                #         //this will hold our custom props for the distance/muscle joints
                #         Json::Value props;
                props = dict()
                #
                #         if (amp < amplitudeCutoff)
                #             this->addDistanceJoint(sourceID, targetID, props);//, {frequencyHz: 3, dampingRatio:.3});
                if amp < amplitude_cutoff:
                    self.add_distance_joint(source_id, target_id, props)
                else:
                    #         else{
                    #
                    #             //these are our hardcoded spring behaviors, could be altered by CPPN, but that seems risky
                    #             props["frequencyHz"] = 3;
                    #             props["dampingRatio"] = .3;
                    props["frequencyHz"] = 3
                    props["dampingRatio"] = 0.3
                    #
                    #             //Phase/Amplitude set by our cppn outputs
                    #             props["phase"] = cppnOutputs[phaseIx].asDouble();
                    props["phase"] = cppn_outputs[phase_ix]
                    #             //JS Version
                    #             //props["amplitude"] = .3f*connectionDistance*amp;
                    #             props["amplitude"] = .3f*amp;
                    props["amplitude"] = 0.3 * amp
                    #
                    #             //need to scale joints based on size of the screen - this is a bit odd, but should help multiple sizes behave the same!
                    #             this->addMuscleJoint(sourceID, targetID, props);
                    self.add_muscle_joint(source_id, target_id, props)
            #         }
            #     }
            #     catch (exception e)
            #     {
            #         printf("Oops error %s", e.what());
            #         throw e;
            #     }
            except Exception as e:
                # why?
                print("Oops error", e)
                raise e
        #
        # }
        #
        # //add the concept of mass
        # morphology["mass"] = morphology["totalNodes"] + connectionDistanceSum/2.0f;
        morphology["mass"] = morphology["totalNodes"] + connection_distance_sum / 2.0
        #
        # //remove the concept of totalNodes since we have mass
        # //morphology should now have width/height, startX/startY, and mass
        # morphology.erase("totalNodes");
        del morphology["totalNodes"]
        #
        # return &morphology;
        return morphology
        # }

    def get_body_entities(
        self, in_body: dict, width_height: b2.b2Vec2, initial_morphology: dict
    ) -> list:
        #
        #
        # std::vector<Json::Value> IESoRWorld::getBodyEntities(Json::Value& inBody, b2Vec2 widthHeight, std::map<std::string, double>& initMorph)
        # {
        #     //We're going to first convert our body into a collection of physics objects that need to be added to the environment
        #     std::vector<Json::Value> entityVector;
        entity_vector = list()
        #
        #     //We'll be using our window width/height for adjusting relative body location
        #     //remember that the CPPN is going to paint information onto a grid, we need to interpret that information according the appropriate sizes
        #     double canvasWidth = widthHeight.x;
        canvas_width = width_height.x
        #     double canvasHeight = widthHeight.y;
        canvas_height = width_height.y
        #
        #     //Grab an array of nodes from our json object
        #     Json::Value oNodes = inBody["nodes"];
        o_nodes = in_body["nodes"]
        #
        #     //this is the starting body identifier for each node (every node has a unique ID for the source connections)
        #     //so node 5 is always node 5, but if you already have objects in the physical world
        #     //object 5 != node 5, so the number of objects + nodeID = true nodeID
        #     int bodyCount = this->world->GetBodyCount();
        body_count = self.world.bodyCount
        #     int bodyID = bodyCount;
        body_id = body_count
        #
        #     bool useLEO = inBody["useLEO"].asBool();
        useLEO = in_body["useLEO"]
        #
        #     //manipulated values of each node, adjusted for screen coorindates
        #     double xScaled, yScaled;
        x_scaled = 0.0
        y_scaled = 0.0
        #
        #     //Adjusting the total size of the object according to window size
        #     double divideForMax = 2.2;
        divide_for_max = 2.2
        #     double divideForMaxHeight = 2.5f;
        divide_for_max_height = 2.5
        #
        #     //Part of a calculation for determinig the x,y values of each node
        #     double maxAllowedWidth = canvasWidth / divideForMax;
        max_allowed_width = canvas_width / divide_for_max
        #     double maxAllowedHeight = canvasHeight / divideForMaxHeight;
        max_allowed_height = canvas_height / divide_for_max_height
        #
        #     //starting values, these will be adjusted
        #     double minX = canvasWidth; double maxX = 0.0f;
        min_x = canvas_width
        max_x = 0.0
        #     double minY = canvasHeight; double maxY = 0.0f;
        min_y = canvas_height
        max_y = 0.0
        #
        #     //We loop through all of our nodes
        #     for (int i=0; i < oNodes.size(); i++)
        for i in range(len(o_nodes)):
            #     {
            #         //for each our nodes, we have a location in the world
            #         Json::Value nodeLocation = oNodes[i];
            node_location = o_nodes[i]
            #
            #         //We pull
            #         double nodeX = nodeLocation["x"].asDouble();
            #         double nodeY = nodeLocation["y"].asDouble();
            node_x = node_location["x"]
            node_y = node_location["y"]
            #
            #         //here we actually modify the x and y values to fit into a certain sized box depending on the initial screen size/physics world size
            #         xScaled = (nodeX)/divideForMax* maxAllowedWidth;
            x_scaled = node_x / divide_for_max * max_allowed_width
            #         yScaled = (1.0f - nodeY) / divideForMaxHeight* maxAllowedHeight;
            y_scaled = (1.0 - node_y) / divide_for_max_height * max_allowed_height
            #
            #         //FOR each node, we make a body with certain properties, then increment count
            #         entityVector.push_back(Entity::CircleEntity(toString(bodyID), xScaled, yScaled, maxAllowedWidth/35.0, 0));
            entity_vector.append(
                Entity.circle_entity(
                    str(body_id), x_scaled, y_scaled, max_allowed_width / 35.0, 0
                )
            )
            #
            #         ///what's the min/max xValues we've seen? We can determine how wide the creature is
            #         minX = min(minX, xScaled);
            min_x = min(min_x, x_scaled)
            #         maxX = max(maxX, xScaled);
            max_x = max(max_x, x_scaled)
            #
            #         ///what's the min/max y values we've seen? We can determine how tall the creature is
            #         minY = min(minY, yScaled);
            min_y = min(min_y, y_scaled)
            #         maxY = max(maxY, yScaled);
            max_y = max(max_y, y_scaled)
            #
            #         //need to increment the body id so we don't overwrite previous object
            #         bodyID++;
            body_id += 1
        #     }
        #
        #     //We use movex, movey to center our creature in world space after creation
        #     double moveX = (maxX - minX) / 2.0;
        move_x = (max_x - min_x) / 2.0
        #     double moveY = (maxY - minY) / 2.0;
        move_y = (max_y - min_y) / 2.0
        #
        #     //Need to move x,y coordinates for entities
        #     //no one should get an unfair advantage being at a certain screen location
        #     for (std::vector<Json::Value>::iterator it = entityVector.begin() ; it != entityVector.end(); ++it)
        #     {
        for e in entity_vector:
            #         Json::Value e = *it;
            #         e["x"] = (e["x"].asDouble() - minX) + canvasWidth/2 - moveX;
            #         e["y"] = (e["y"].asDouble() - minY) + canvasHeight/2 - moveY;
            e["x"] = (float(e["x"]) - min_x) + canvas_width / 2 - move_x
            e["y"] = (float(e["y"]) - min_y) + canvas_height / 2 - move_y
        #     }
        #
        #     //We now have everything we need to identify our initial morphology
        #     //we can see the inital width/height of the object
        #     //as well as the offset (startx, starty) of the object
        #     initMorph["width"] = maxX - minX;
        initial_morphology["width"] = max_x - min_x
        #     initMorph["height"] = maxY - minY;
        initial_morphology["height"] = max_y - min_y
        #     initMorph["startX"] = minX;
        initial_morphology["startX"] = min_x
        #     initMorph["startY"] = minY;
        initial_morphology["startY"] = min_y
        #     initMorph["totalNodes"] = oNodes.size();
        initial_morphology["totalNodes"] = len(o_nodes)
        #
        #     return entityVector;
        # }
        return entity_vector

    def set_bodies(self, entities: list) -> None:
        #
        # //pretty simple, we loop through and call the set body function that does all the work
        # void IESoRWorld::setBodies(vector<Json::Value>* entities)
        # {
        #     //look through our vector of entities, and add each entity to the world
        #     for (std::vector<Json::Value>::iterator it = entities->begin() ; it != entities->end(); ++it)
        #     {
        #         Json::Value e = *it;
        #         string bodyID = e["bodyID"].asString();
        #
        #         this->setBody(e);
        #     }
        # }
        for e in entities:
            body_id = e["bodyID"]
            self.set_body(e)
        return

    def load_data_file(self, data_name: str) -> str:
        #
        # //laod object from data folder -- simple!!
        # std::string IESoRWorld::loadDataFile(std::string dataName)
        # {
        #     std::string filePath = "../../../Physics/Data/" + dataName;
        #     return get_file_contents(filePath.c_str());
        # }

        # TODO: figure out where this should be located. For now this is fine.
        physics_datapath = "Physics/Data"
        return get_file_contents(os.path.join(physics_datapath, data_name))

    def add_distance_joint(self, source_id: str, target_id: str, props: dict) -> Bone:
        #
        #
        # Bone* IESoRWorld::addDistanceJoint(std::string sourceID, std::string targetID, Json::Value props)
        # {
        # //we're connecting two body objects together
        # //they should be defined by their IDs in our bodymap
        # b2Body* body1 = this->bodyMap[sourceID];
        # b2Body* body2 = this->bodyMap[targetID];
        body1 = self.body_map[source_id]  # type: b2.b2Body
        body2 = self.body_map[target_id]  # type: b2.b2Body
        #
        # //Create a basic distance joint
        # b2DistanceJointDef* joint = new b2DistanceJointDef();
        joint = b2.b2DistanceJointDef()
        #
        # //initialize the joints where they're attached at the center of the objects
        # joint->Initialize(body1, body2, body1->GetWorldCenter(), body2->GetWorldCenter());
        joint.Initialize(body1, body2, body1.worldCenter, body2.worldCenter)
        #
        # //check if we defined frequencyHz value, if so fetch it from json
        # //this has to do with how the distance joint responds to stretching
        # if(!props["frequencyHz"].isNull())
        # {
        #     joint->frequencyHz = props["frequencyHz"].asDouble();
        # }
        if props.get("frequencyHz", None) is not None:
            joint.frequencyHz = props["frequencyHz"]
        #
        # //check if we defined dampingRatio value, if so fetch it from json -- check docs for dampingRatio usage
        # if(!props["dampingRatio"].isNull())
        # {
        #     joint->dampingRatio = props["dampingRatio"].asDouble();
        # }
        if props.get("dampingRatio", None) is not None:
            joint.dampingRatio = props["dampingRatio"]
        # else:
        #     joint.dampingRatio = 1.0
        joint.collideConnected = True
        #
        # //let's use our definition to make a real live joint!
        # b2Joint* wJoint = world->CreateJoint(joint);
        w_joint = self.world.CreateJoint(joint)  # type: b2.b2DistanceJoint
        # w_joint.collideConnected
        #
        # //we identify our bones by the count in the list
        # Bone* bone = new Bone(toString(boneList.size()), wJoint);
        bone = Bone(str(len(self.bone_list)), w_joint)
        #
        # //we push our joint into a list of joints created
        # boneList.push_back(bone);
        self.bone_list.append(bone)
        #
        # //we store this distance joint info inside of the joint!
        # wJoint->SetUserData(bone);
        w_joint.userData = bone
        #
        # //finished, return our bone object created
        # return bone;
        # }
        #
        #
        return bone

    def add_muscle_joint(self, source_id: str, target_id: str, props: dict) -> Muscle:
        # Muscle* IESoRWorld::addMuscleJoint(std::string sourceID, std::string targetID, Json::Value props)
        # {
        # //we add a standard distance joint
        # Bone* addedJoint = addDistanceJoint(sourceID, targetID, props);
        added_joint = self.add_distance_joint(source_id, target_id, props)
        #
        # //But our muscles have phase and amplitude, which adjust length during updates
        # double phase = 0;
        phase = 0
        #
        # //check if we defined phase value, if so fetch it from json
        # if(!props["phase"].isNull())
        # {
        #     phase = props["phase"].asDouble();
        # }
        if props.get("phase", None) is not None:
            phase = props["phase"]
        #
        # //amplitude is how much change each muscles exhibits
        # double amplitude = 1;
        #
        # //check if it's in our props object
        # if(!props["amplitude"].isNull())
        # {
        #     phase = props["amplitude"].asDouble();
        # }
        amplitude = 1 if props.get("amplitude", None) is None else props["amplitude"]
        #
        # //muscle is just a container for our
        # Muscle* ms = new Muscle(toString(this->muscleList.size()), addedJoint->GetJoint(), amplitude, phase);
        ms = Muscle(str(len(self.muscle_list)), added_joint.joint, amplitude, phase)
        #
        # //track the muscle objects this way, pretty easy
        # this->muscleList.push_back(ms);
        self.muscle_list.append(ms)
        #
        # //send back the joint we added to physical world
        # return ms;
        return ms

    def set_body(self, entity: dict) -> None:
        #
        #
        # void IESoRWorld::setBody(Json::Value& entity)
        # {
        # //we create a generic body def, to hold our body
        # b2BodyDef* bodyDef = new b2BodyDef();
        body_def = b2.b2BodyDef()
        #
        # //we are either adding "ground" (static) or we're adding everything else (dynamic)
        # if (entity["bodyID"].asString() == "ground")
        if entity["bodyID"] == "ground":
            # {
            #     bodyDef->type = b2BodyType::b2_staticBody;
            body_def.type = b2.b2_staticBody
        # }
        # else
        else:
            # {
            #     bodyDef->type = b2BodyType::b2_dynamicBody;
            body_def.type = b2.b2_dynamicBody
        # }
        #
        # //we set the position to the intial x,y coordinate
        # bodyDef->position.Set(entity["x"].asDouble(), entity["y"].asDouble());
        body_def.position = b2.b2Vec2(entity["x"], entity["y"])
        #
        # //we set physical properties of this body
        # //we don't want any rotation on the nodes (they would slide when moving)
        # bodyDef->linearDamping = 1.1f;
        body_def.linearDamping = 1.1
        #
        # //bodies have fixed rotation in iesor -- no move please k thx
        # bodyDef->fixedRotation = true;
        body_def.fixedRotation = True
        #
        # if(!entity["angle"].isNull())
        #     bodyDef->angle = entity["angle"].asDouble();
        if entity.get("angle", None) is not None:
            body_def.angle = entity["angle"]
        #
        # //we add our body to the world, and register it's identifying information
        # //that's useful for drawing
        # b2Body* body = this->addBodyToWorld(entity["bodyID"].asString(), bodyDef);
        body = self.add_body_to_world(entity["bodyID"], body_def)
        #
        # //we check if we have a radius (that makes us a circle entity)
        # if (!entity["radius"].isNull())
        if entity.get("radius", None) is not None:
            # {
            #     //this filter prevents circle nodes from colliding on our bodies
            #     b2Filter* filter = new b2Filter();
            filter = b2.b2Filter()
            #     filter->categoryBits = 0x0002;
            filter.categoryBits = 0x0002
            #     filter->maskBits = 0x0001;
            filter.maskBits = 0x0001
            #     filter->groupIndex = 0;
            filter.groupIndex = 0
            #
            #     //we create a fixture def for our new shape
            #     b2FixtureDef fixture;
            fixture = b2.b2FixtureDef()
            #
            #     //We need high density objects so they don't float!
            #     fixture.density = 25.0;
            fixture.density = 25.0
            #     //lots of friction please!
            #     fixture.friction = 1.0;
            fixture.friction = 1.0
            #     //less bouncing, more staying
            #     fixture.restitution = .1;
            fixture.restitution = 0.1
            #
            #     //we're a circle shape, make it so
            #     b2CircleShape node;
            node = b2.b2CircleShape()
            #
            #     //and we establish our radius
            #     node.m_radius = entity["radius"].asDouble();
            node.radius = entity["radius"]
            #
            #     //attach the shape to our fixtureDef
            #     fixture.shape = &node;
            fixture.shape = node
            #     fixture.filter = *filter;
            fixture.filter = filter
            #
            #     //add this fixture to our body (and track it in this class)
            #     this->addShapeToBody(body, &fixture);
            self.add_shape_to_body_fixture(body, fixture)
        # }
        # //now we're checking if we're a polygonal shape
        # else if (!entity["polyPoints"].isNull())
        elif entity.get("polyPoints", None) is not None:
            # {
            #     //we know how many points we need to add
            #     int pointCount = entity["polyPoints"].size();
            point_count = len(entity["polyPoints"])
            #
            #     //so we create an array to hold those points in a form Box2D undertstands
            #     b2Vec2* polyPoints = new b2Vec2[pointCount];
            #     //loop through our json points
            #     for (int j = 0; j < pointCount; j++)
            #     {
            #         //pull the object from our json array
            #         Json::Value point = entity["polyPoints"][j];
            #
            #         //convert to box2d coords
            #         b2Vec2 vec(point["x"].asDouble(), point["y"].asDouble());
            #
            #         //set our coordinate inside the array
            #         polyPoints[j] = vec;
            #     }
            poly_points = [
                b2.b2Vec2(point["x"], point["y"]) for point in entity["polyPoints"]
            ]
            #
            #     //create our polygon shape
            #     b2PolygonShape poly;
            poly = b2.b2PolygonShape()
            #
            #     //set our box2d points as the polygonal shape
            #     poly.Set(polyPoints, pointCount);
            poly.vertices = poly_points
            #
            #     //defalult density to 0, and add the sucker
            #     this->addShapeToBody(body, &poly, 0.0);
            self.add_shape_to_body(body, poly, 0.0)
        # }
        # else
        else:
            # {
            #     //we're a rectangle entity!
            #     b2PolygonShape poly;
            poly = b2.b2PolygonShape
            #
            #     //we've got a box, we simply put in the half height/ hafl width info for our polygon
            #     //it does the rest
            #     poly.SetAsBox(entity["halfWidth"].asDouble(),entity["halfHeight"].asDouble());
            poly.SetAsBox(entity["halfWidth"], entity["halfHeight"])
            #
            #     //defalult density to 0, and add the sucker
            #     this->addShapeToBody(body, &poly, 0.0);
            self.add_shape_to_body(body, poly, 0.0)
        # }
        #
        # //we've created our body and our shape, and it's all registered in the class
        # }
        return

    def create_shape_id(self) -> PhysicsId:
        # PhysicsID* IESoRWorld::createShapeID()
        # {
        #     //We just keep a static int of all the shapes for all identification
        #     int bcnt = staticBodyCount++;
        #
        #     //Track the shape string
        #     PhysicsID* shapeID = new PhysicsID(toString(bcnt));
        #
        #     //Save our shape to the list of shapes
        #     this->shapeList.push_back(shapeID);
        #
        #     return shapeID;
        # }
        global staticBodyCount
        staticBodyCount = staticBodyCount + 1
        shape_id = PhysicsId(str(staticBodyCount))
        self.shape_list.append(shape_id)
        return shape_id


def get_file_contents(filename: str) -> str:
    with open(filename) as f:
        return f.read()


def set_shape_id(fix: b2.b2Fixture, shape_id: PhysicsId):
    fix.userData = shape_id


def position_to_json_value(vec: b2.b2Vec2) -> dict:
    #
    # //converts a b2Vec point into a json object
    # Json::Value positionToJSONValue(b2Vec2 vec)
    # {
    #     Json::Value pos;
    #     pos["x"] = vec.x;
    #     pos["y"] = vec.y;
    #     return pos;
    # }
    return {"x": vec.x, "y": vec.y}


def list_of_points(points: list[b2.b2Vec2], length: int = 0) -> list:
    # //converts an array of points into a json array
    # Json::Value listOfPoints(b2Vec2* points, int length)
    # {
    #     Json::Value pArray(Json::arrayValue);
    #
    #     for(int i=0;i<length;i++)
    #     {
    #         pArray[i] = positionToJSONValue(points[i]);
    #     }
    #
    #     return pArray;
    # }

    # Pretty sure we don't need length...
    # Assuming it works.
    return [position_to_json_value(points[i]) for i in range(length)]


if __name__ == "__main__":
    main(IESoRWorld)
