// TestZoneEvents.cs — DELETE after verifying Step 06
using UnityEngine;

public class TestZoneEvents : MonoBehaviour
{
    [SerializeField] private DestinationZone[] _zones;

    private int _correctCount;
    private int _incorrectCount;
    private int _missedCount;

    private void Start()
    {
        foreach (var z in _zones)
        {
            z.OnCorrectSort += pkg =>
            {
                _correctCount++;
                Debug.Log($"<color=green>CORRECT #{_correctCount}</color>: " +
                          $"{pkg.DestinationLabel} entered {z.name} (accepted: {z.AcceptedLabel})");
            };
            z.OnIncorrectSort += pkg =>
            {
                _incorrectCount++;
                Debug.Log($"<color=red>INCORRECT #{_incorrectCount}</color>: " +
                          $"{pkg.DestinationLabel} entered {z.name} (accepted: {z.AcceptedLabel})");
            };
            z.OnMissedPackage += pkg =>
            {
                _missedCount++;
                Debug.Log($"<color=yellow>MISSED #{_missedCount}</color>: " +
                          $"{pkg.DestinationLabel} fell through {z.name}");
            };
        }
    }
}