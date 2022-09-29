import Box2D.Box2D as b2
from enum import Enum
from typing import List

# omg globals 🤮
# TODO: Figure out a better way than a global, otherwise my code standards guy will be more disappointed in me.
staticBodyCount: int = 0


class EntityType(Enum):
    Circle = 0
    Polygon = 1
    Rectangle = 2



class Entity:
    @staticmethod
    def circle_entity(body_id: str, x: float, y: float, radius: float, angle: float) -> dict:
        # static Json::Value CircleEntity(std::string bodyID, float x, float y, float radius, float angle)
        # {
        #     Json::Value e;
        #     e["type"] = EntityType::Circle;
        #     e["bodyID"] = bodyID;
        #     e["x"] = x;
        #     e["y"] = y;
        #     e["angle"] = angle;
        #     e["radius"] = radius;
        #     return e;
        # }
        e = dict()
        e["type"] = EntityType.Circle
        e["bodyID"] = body_id
        e["x"] = x
        e["y"] = y
        e["angle"] = angle
        e["radius"] = radius
        return e
    
    @staticmethod
    def polygon_entity(body_id: str, x_scale: float, y_scale: float, points: List[b2.b2Vec2], angle: float) -> dict:
        # static Json::Value PolygonEntity(std::string bodyID, float xScale, float yScale, std::vector<b2Vec2>* points, float angle)
        # {
        #     Json::Value e;
        # 
        #     e["type"] = EntityType::Polygon;
        #     e["bodyID"] = bodyID;
        #     e["x"] = xScale;
        #     e["y"] = yScale;
        #     e["angle"] = angle;
        # 
        #     //map the points in please!
        #     Json::Value polyPoints(Json::arrayValue);
        #     int ix = 0;
        # 
        #     //add all the poitns into our new point vector!
        #     for (std::vector<b2Vec2>::iterator it = points->begin() ; it != points->end(); ++it)
        #     {
        #         Json::Value point; 
        #         point["x"] = it->x;
        #         point["y"] = it->y;
        #         polyPoints[ix++] = point;
        #     }
        # 
        #     return e;
        # }
        e = dict()
        e["type"] = EntityType.Polygon
        e["bodyID"] = body_id
        e["x"] = x_scale
        e["y"] = y_scale
        e["angle"] = angle
        e["polyPoints"] = list()
        for point in points:
            e["polyPoints"].append(
                {'x': point.x, 'y': point.y}
            )
        return e
    
    @staticmethod
    def rectangle_entity(body_id: str, x_scale: float, y_scale: float, 
                         half_width: float, half_height: float, angle: float):
        # static Json::Value RectangleEntity(std::string bodyID, float xScale, float yScale, float halfWidth, float halfHeight, float angle)
        # {             
        #     Json::Value e;
        # 
        #     e["type"] = EntityType::Rectangle;
        #     e["bodyID"] = bodyID;
        #     e["x"] = xScale;
        #     e["y"] = yScale;
        #     e["angle"] = angle;
        #     e["halfWidth"] = halfWidth;
        #     e["halfHeight"] = halfHeight;
        # 
        #     return e;
        # }
        e = dict()
        e['type'] = EntityType.Rectangle
        e['bodyID'] = body_id
        e['x'] = x_scale
        e['y'] = y_scale
        e['angle'] = angle
        e['halfWidth'] = half_width
        e['halfHeight'] = half_height

class BodyInformation:
    # {
    # public:
    #     BodyInformation();
    #     //Keep an ID present for each body chunk
    #     std::string GenomeID;
    #
    #     //Required Nodes inside the structure
    #     std::vector<b2Vec2> nodes;
    #
    #     //All connections required for the sodaracer
    #     std::vector<b2Vec2> connections;
    #
    #     //this is for mapping between an index and a
    #     //connection number (required for pruning)
    #     std::map<int,int> indexToConnection;
    #
    #     //Is LEO used in the generation of this body
    #     //(important for output tracking)
    #     bool useLEO;
    #
    # private:
    #     ~BodyInformation();
    # };
    def __init__(self):
        self.genome_id: str = ""
        self.nodes = list()
        self.connections = list()
        self.index_to_connection = dict()
        self.useLEO = False



class PhysicsId:
    def __init__(self, pid: str):
        self.__id = pid

    def id(self):
        return self.__id


class DistanceAccessor(b2.b2DistanceJoint):
    # It was either this or access the name mangled
    # __getLength from the b2DistanceJoint so...
    getLength = b2._swig_new_instance_method(
        b2._Box2D.b2DistanceJoint___GetLength
    )
    setLength = b2._swig_new_instance_method(
        b2._Box2D.b2DistanceJoint___SetLength
    )


class Muscle:
    def __init__(self, muscle_id: str, joint: b2.b2Joint, phase: float, amplitude: float):
        # store the id, joint, phase, and amplitude -- pretty simple!
        self.muscle_id = muscle_id
        self.joint = joint
        self.phase = phase
        self.amplitude = amplitude

        # pull the original length from our distance joint
        self.o_length = DistanceAccessor.getLength(joint)  # ((b2DistanceJoint*)joint)->GetLength();

    # getters to retrieve inner variables
    def id(self) -> str:
        return self.muscle_id

    def get_joint(self) -> b2.b2Joint:
        return self.joint

    def get_original_length(self) -> float:
        return self.o_length

    def get_phase(self) -> float:
        return self.phase

    def get_amplitude(self) -> float:
        return self.amplitude


class Bone:
    def __init__(self, muscle_id: str, joint: b2.b2Joint):
        self.bone_id = muscle_id
        self.joint = joint

    def get_joint(self) -> b2.b2Joint:
        return self.joint

    def id(self) -> str:
        return self.bone_id
