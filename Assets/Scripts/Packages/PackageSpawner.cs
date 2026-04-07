using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Spawns packages with Poisson-distributed inter-arrival times.
/// Uses a pre-allocated object pool — no Instantiate() or Destroy() during episodes.
/// </summary>
public class PackageSpawner : MonoBehaviour
{
    // ── Configuration ──────────────────────────────────────────────
    [Header("Spawn Settings")]
    [Tooltip("Average arrivals per minute (λ). Default 30 = one every 2s on average.")]
    [SerializeField] private float _arrivalsPerMinute = 30f;

    [Tooltip("Maximum lateral offset (±) from spawn centre, in meters.")]
    [SerializeField] private float _lateralOffsetMax = 0.1f;

    [Header("Pool Settings")]
    [Tooltip("Total packages pre-allocated in the pool.")]
    [SerializeField] private int _poolSize = 40;

    [Header("Package Types")]
    [Tooltip("Assign all 3 PackageTypeSO assets here.")]
    [SerializeField] private PackageTypeSO[] _packageTypes;

    [Header("References")]
    [Tooltip("Empty transform marking the spawn position.")]
    [SerializeField] private Transform _spawnPoint;

    // ── Pool ────────────────────────────────────────────────────────
    private readonly List<Package> _pool = new List<Package>();
    private GameObject _prefab;

    // ── Timing ──────────────────────────────────────────────────────
    private float _nextSpawnTime;

    // ── Stats ───────────────────────────────────────────────────────
    /// <summary>Total packages spawned this episode.</summary>
    public int TotalSpawned { get; private set; }

    private void Awake()
    {
        Debug.Assert(_packageTypes != null && _packageTypes.Length == 3,
            "[PackageSpawner] Exactly 3 PackageTypeSO assets required.");
        Debug.Assert(_spawnPoint != null,
            "[PackageSpawner] _spawnPoint not assigned.");
        Debug.Assert(_poolSize > 0,
            "[PackageSpawner] _poolSize must be positive.");
        Debug.Assert(_arrivalsPerMinute > 0f,
            "[PackageSpawner] _arrivalsPerMinute must be positive.");

        CreatePool();
    }

    private void Start()
    {
        _nextSpawnTime = Time.fixedTime + SamplePoissonInterval();
    }

    private void FixedUpdate()
    {
        if (Time.fixedTime >= _nextSpawnTime)
        {
            TrySpawnPackage();
            _nextSpawnTime = Time.fixedTime + SamplePoissonInterval();
        }
    }

    // ── Pool management ─────────────────────────────────────────────
    private void CreatePool()
    {
        // Create a runtime prefab template (never activated in scene)
        _prefab = GameObject.CreatePrimitive(PrimitiveType.Cube);
        _prefab.name = "PackageTemplate";
        _prefab.SetActive(false);

        // Add required components
        if (_prefab.GetComponent<Rigidbody>() == null)
            _prefab.AddComponent<Rigidbody>();
        if (_prefab.GetComponent<Package>() == null)
            _prefab.AddComponent<Package>();
        if (_prefab.GetComponent<PackageConveyor>() == null)
            _prefab.AddComponent<PackageConveyor>();

        // Pre-allocate pool
        for (int i = 0; i < _poolSize; i++)
        {
            GameObject obj = Instantiate(_prefab, transform);
            obj.name = $"Package_{i:D2}";
            obj.SetActive(false);
            _pool.Add(obj.GetComponent<Package>());
        }

        // Hide template (not destroyed — no Destroy during runtime)
        _prefab.transform.SetParent(transform);
    }

    private Package GetFromPool()
    {
        for (int i = 0; i < _pool.Count; i++)
        {
            if (!_pool[i].gameObject.activeSelf)
                return _pool[i];
        }

        Debug.LogWarning("[PackageSpawner] Pool exhausted — all packages active.");
        return null;
    }

    /// <summary>
    /// Return all active packages to pool. Called by EnvironmentManager on episode reset.
    /// </summary>
    public void ReturnAllToPool()
    {
        for (int i = 0; i < _pool.Count; i++)
        {
            if (_pool[i].gameObject.activeSelf)
            {
                Rigidbody rb = _pool[i].GetComponent<Rigidbody>();
                rb.velocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                _pool[i].ReturnToPool();
            }
        }

        TotalSpawned = 0;
        _nextSpawnTime = Time.fixedTime + SamplePoissonInterval();
        
    }

    // ── Spawning ────────────────────────────────────────────────────
    private void TrySpawnPackage()
    {
        Package pkg = GetFromPool();
        if (pkg == null) return;

        

        // Random type
        PackageTypeSO type = _packageTypes[Random.Range(0, _packageTypes.Length)];
        pkg.gameObject.SetActive(true);

        // Position: spawn point + random lateral offset on local X axis
        float lateralOffset = Random.Range(-_lateralOffsetMax, _lateralOffsetMax);
        Vector3 pos = _spawnPoint.position + _spawnPoint.right * lateralOffset;
        // Raise slightly above belt to avoid spawning inside it
        pos.y = _spawnPoint.position.y + type.dimensions * 0.5f + 0.05f;

        pkg.transform.position = pos;
        pkg.transform.rotation = Quaternion.identity;
        pkg.Initialise(type);

        TotalSpawned++;
    }

    // ── Poisson sampling ────────────────────────────────────────────
    /// <summary>
    /// Sample inter-arrival time from exponential distribution.
    /// For Poisson process with rate λ, inter-arrival times are Exp(1/λ).
    /// </summary>
    private float SamplePoissonInterval()
    {
        float lambda = _arrivalsPerMinute / 60f; // convert to per-second
        Debug.Assert(lambda > 0f, "[PackageSpawner] λ must be positive.");

        // Inverse CDF of exponential: -ln(U) / λ, where U ~ Uniform(0,1)
        // Clamp U away from 0 to avoid ln(0) = -infinity
        float u = Random.Range(0.0001f, 1f);
        return -Mathf.Log(u) / lambda;
    }
}