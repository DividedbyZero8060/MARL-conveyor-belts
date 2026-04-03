using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;


public class AgentPlaceholder : Agent
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public override  void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Do nothing
    }

}
