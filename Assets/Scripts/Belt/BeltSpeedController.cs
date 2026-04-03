using System;
using UnityEngine;

/// <summary>
/// Global belt speed controller.  One instance in the scene (singleton pattern).
/// Exposes <see cref="CurrentSpeed"/> (m/s) clamped to [0, <see cref="MaxSpeed"/>]
/// and fires <see cref="OnSpeedChanged"/> whenever the value changes.
/// </summary>
public class BeltSpeedController : MonoBehaviour
{
    // ── Constants ────────────────────────────────────────────────────
    private const float MIN_SPEED = 0f;

    [Header("Speed Settings")]
    [Tooltip("Maximum allowed belt speed in m/s.")]
    [SerializeField] private float _maxSpeed = 5f;

    [Tooltip("Initial belt speed in m/s.")]
    [SerializeField] private float _initialSpeed = 2f;

    // ── Runtime state ───────────────────────────────────────────────
    private float _currentSpeed;

    /// <summary>Current belt speed in m/s, clamped [0, MaxSpeed].</summary>
    public float CurrentSpeed
    {
        get => _currentSpeed;
        set
        {
            float clamped = Mathf.Clamp(value, MIN_SPEED, _maxSpeed);
            if (Mathf.Approximately(clamped, _currentSpeed)) return;
            _currentSpeed = clamped;
            OnSpeedChanged?.Invoke(_currentSpeed);
        }
    }

    /// <summary>Upper speed limit (read-only at runtime).</summary>
    public float MaxSpeed => _maxSpeed;

    /// <summary>Fired whenever <see cref="CurrentSpeed"/> changes. Argument = new speed.</summary>
    public event Action<float> OnSpeedChanged;

    // ── Singleton (lightweight, no DontDestroyOnLoad) ───────────────
    public static BeltSpeedController Instance { get; private set; }

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning($"[BeltSpeedController] Duplicate on {name}; destroying.");
            Destroy(this);
            return;
        }
        Instance = this;

        Debug.Assert(_maxSpeed > 0f,
            "[BeltSpeedController] _maxSpeed must be positive.");
        Debug.Assert(_initialSpeed >= MIN_SPEED && _initialSpeed <= _maxSpeed,
            "[BeltSpeedController] _initialSpeed out of range.");

        _currentSpeed = Mathf.Clamp(_initialSpeed, MIN_SPEED, _maxSpeed);
    }

    // ── Editor convenience: slider in Play mode ─────────────────────
    private void OnValidate()
    {
        if (Application.isPlaying)
        {
            CurrentSpeed = _initialSpeed;
        }
    }
}