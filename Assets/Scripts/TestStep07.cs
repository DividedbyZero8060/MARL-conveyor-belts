using UnityEngine;

public class TestStep07 : MonoBehaviour
{
    [SerializeField] private DiverterGate _gate1;
    [SerializeField] private DiverterGate _gate2;
    [SerializeField] private DiverterGate _gate3;

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Q)) Debug.Log($"Gate1 Activate: {_gate1.Activate()} (state={_gate1.CurrentState})");
        if (Input.GetKeyDown(KeyCode.W)) Debug.Log($"Gate2 Activate: {_gate2.Activate()} (state={_gate2.CurrentState})");
        if (Input.GetKeyDown(KeyCode.E)) Debug.Log($"Gate3 Activate: {_gate3.Activate()} (state={_gate3.CurrentState})");
        if (Input.GetKeyDown(KeyCode.R))
        {
            EnvironmentManager.Instance.ResetEpisode();
            Debug.Log($"Reset #{EnvironmentManager.Instance.EpisodeIndex}. Mapping: " +
                      $"B0={EnvironmentManager.Instance.GetDestinationForBranch(0)}, " +
                      $"B1={EnvironmentManager.Instance.GetDestinationForBranch(1)}, " +
                      $"B2={EnvironmentManager.Instance.GetDestinationForBranch(2)}");
        }

        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            BeltSpeedController.Instance.CurrentSpeed = 1.0f;
            Debug.Log("Belt speed set to 1.0");
        }
        if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            BeltSpeedController.Instance.CurrentSpeed = 5.0f;
            Debug.Log("Belt speed set to 5.0");
        }
        if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            BeltSpeedController.Instance.CurrentSpeed = 0.0f;
            Debug.Log("Belt speed set to 0.0");
        }
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log($"Gate1: state={_gate1.CurrentState}, " +
                      $"actionable={_gate1.IsActionable}, " +
                      $"cooldown={_gate1.NormalisedCooldownRemaining:F2}");
        }
    }
}